# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
import socket

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

import pandas as pd
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow_config import *

from config import get_config
# from models import build_model
from models.teacher import build_simmim_testing
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, write_epoch_to_csv

from pathlib import Path

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None



# Optimize PyTorch precision
torch.set_float32_matmul_precision('medium')

def setup_mlflow_function():
    # Set MLflow request timeout
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "3500"
    print(f" MLflow Tracking URI Set: {mlflow.get_tracking_uri()}")

    # Initialize MLflow Client
    client = MlflowClient()

    # Define Experiment Name
    experiment_name = "GFMaerial_swin_teacher"

    # Check if the Experiment Exists
    experiment = client.get_experiment_by_name(experiment_name)

    # Restore if it was soft-deleted
    if experiment and experiment.lifecycle_stage == "deleted":
        print(f" Experiment '{experiment_name}' is soft-deleted. Restoring...")
        client.restore_experiment(experiment.experiment_id)
        experiment = client.get_experiment_by_name(experiment_name)

    # Create new if not found
    if experiment is None:
        print(f" Experiment '{experiment_name}' not found! Creating a new one...")
        experiment_id = client.create_experiment(name=experiment_name)
        print(f" Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f" Using existing experiment: {experiment_name} (ID: {experiment_id})")

    # Set the active experiment
    mlflow.set_experiment(experiment_name)

    # Confirm Artifact Location
    experiment = client.get_experiment(experiment_id)
    print(f" Experiment '{experiment_name}' Artifact Location: {experiment.artifact_location}")
    return

def mlflow_logging_workflow(log_path):
    logger.info("mlflow_logging:")
    try:
        setup_mlflow_function()
        logger.info("Mlflow setup successfull")
        with mlflow.start_run(run_name=config.MODEL.NAME):
            # Add tags for traceability
            mlflow.set_tag("mlflow.user", socket.gethostname())  # Will show as "Created by"
            mlflow.set_tag("mlflow.source.name", os.path.abspath(__file__))  # Will show as "Source"
            mlflow.log_params({
                "model": config.MODEL.NAME,
                "dataset": config.DATA.DATA_TRAIN_PATH,
                "epochs": config.TRAIN.EPOCHS,
                "batch_size": config.DATA.BATCH_SIZE,
                "accum_steps": config.TRAIN.ACCUMULATION_STEPS,
                "alpha": config.ALPHA,
                "base_lr": config.TRAIN.BASE_LR
            })

            mlflow.log_artifact(path)

            # Call training inside the same run
            logger.info(f"start logging")

            for file in os.listdir(log_path):
                if file.endswith(".csv"):
                    df = pd.read_csv(Path(log_path) / file, on_bad_lines='skip')
                    filename = Path(file).stem
                    if "table" in filename:
                        mlflow.log_table(df, artifact_file=f"{filename}.json")

                    elif "track" in filename:
                        for _, row in df.iterrows():
                            if pd.isna(row['epoch']) or pd.isna(row[filename]):
                                continue
                            mlflow.log_metric(filename, float(row[filename]), step=int(row["epoch"]))
                    else:
                        continue

    except Exception as e:
        logger.info(f"mlflow not available. Logging interrupted by: {e}")
        return

    return


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha for similarity loss")

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    data_loader_vali_temp_ind = build_loader(config, logger, is_pretrain=True, is_train=False, vali_key=0)
    data_loader_vali_spa_ind = build_loader(config, logger, is_pretrain=True, is_train=False, vali_key=1)
    data_loader_vali_temp_spa_ind = build_loader(config, logger, is_pretrain=True, is_train=False, vali_key=2)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_simmim_testing(config, logger)
    model.cuda()
    logger.info(str(model))
    #logger.info(torch.cuda.is_available())
    #logger.info(torch.backends.cudnn.enabled)
    #logger.info(f"device of model: {next(model.parameters()).device}")

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    logger.info("Start testing")
    start_time = time.time()

    loss_log_header = ["avg_l1_rgbi", "avg_l2_rgbi", "avg_l1_rgb","avg_l2_rgb"]

    l1_rgbi_temp_ind, l2_rgbi_temp_ind, l1_rgb_temp_ind, l2_rgb_temp_ind = test_generalization(config, model, data_loader_vali_temp_ind, test_key="temp_ind")
    l1_rgbi_spa_ind, l2_rgbi_spa_ind, l1_rgb_spa_ind, l2_rgb_spa_ind = test_generalization(config, model, data_loader_vali_spa_ind, test_key="spa_ind")
    l1_rgbi_temp_spa_ind, l2_rgbi_temp_spa_ind, l1_rgb_temp_spa_ind, l2_rgb_temp_spa_ind = test_generalization(config, model, data_loader_vali_temp_spa_ind, test_key="temp_spa_ind")
    avg_l1_rgbi = (l1_rgbi_temp_ind + l1_rgbi_spa_ind+l1_rgbi_temp_spa_ind)/3
    avg_l2_rgbi = (l2_rgbi_temp_ind + l2_rgbi_spa_ind+l2_rgbi_temp_spa_ind)/3
    avg_l1_rgb = (l1_rgb_temp_ind + l1_rgb_spa_ind+l1_rgb_temp_spa_ind)/3
    avg_l2_rgb = (l2_rgb_temp_ind + l2_rgb_spa_ind+l2_rgb_temp_spa_ind)/3

    avg_log_loss = [avg_l1_rgbi, avg_l2_rgbi, avg_l1_rgb,avg_l2_rgb]

    write_epoch_to_csv(os.path.join(config.OUTPUT_STATS, "avg_test_loss_log_table.csv"), avg_log_loss, loss_log_header)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if dist.get_rank() == 0:
        mlflow_logging_workflow(config.OUTPUT_STATS)

    logger.info('Testing time {}'.format(total_time_str))

@torch.no_grad()
def test_generalization(config, model, data_loader, lmdb_key=True, test_key="spa_ind"):
    model.eval()

    batch_time = AverageMeter()
    l1_recon_loss_rgbi_meter = AverageMeter()
    l2_recon_loss_rgbi_meter = AverageMeter()
    l1_recon_loss_rgb_meter = AverageMeter()
    l2_recon_loss_rgb_meter = AverageMeter()

    test_header = ["l1_recon_loss_rgbi_meter_val",
                   "l1_recon_loss_rgbi_meter_avg",
                   "l2_recon_loss_rgbi_meter_val",
                   "l2_recon_loss_rgbi_meter_avg",
                   "l1_recon_loss_rgb_meter_val",
                   "l1_recon_loss_rgb_meter_avg",
                   "l2_recon_loss_rgb_meter_val",
                   "l2_recon_loss_rgb_meter_avg",
                   "batch_time_avg",
                   "batch_time_val",
                   "memory_used",
                   "testing_time"]

    start = time.time()
    end = time.time()
    num_steps = len(data_loader)

    for idx, batch in enumerate(data_loader):
        if lmdb_key:
            img, mask = batch
        else:
            img, mask, _ = batch

        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        l1_recon_loss_rgbi, l2_recon_loss_rgbi, l1_recon_loss_rgb, l2_recon_loss_rgb = model(img, mask)

        torch.cuda.synchronize()

        l1_recon_loss_rgbi_meter.update(l1_recon_loss_rgbi.item(), img.size(0))
        l2_recon_loss_rgbi_meter.update(l2_recon_loss_rgbi.item(), img.size(0))
        l1_recon_loss_rgb_meter.update(l1_recon_loss_rgb.item(), img.size(0))
        l2_recon_loss_rgb_meter.update(l2_recon_loss_rgb.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Test: [{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {l1_recon_loss_rgbi_meter.val:.4f} ({l1_recon_loss_rgbi_meter.avg:.4f})\t'
                f'loss {l2_recon_loss_rgbi_meter.val:.4f} ({l2_recon_loss_rgbi_meter.avg:.4f})\t'
                f'loss {l1_recon_loss_rgb_meter.val:.4f} ({l1_recon_loss_rgb_meter.avg:.4f})\t'
                f'loss {l2_recon_loss_rgb_meter.val:.4f} ({l2_recon_loss_rgb_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    testing_time = time.time() - start
    curr_list = [l1_recon_loss_rgbi_meter.val,
                 l1_recon_loss_rgbi_meter.avg,
                 l2_recon_loss_rgbi_meter.val,
                 l2_recon_loss_rgbi_meter.avg,
                 l1_recon_loss_rgb_meter.val,
                 l1_recon_loss_rgb_meter.avg,
                 l2_recon_loss_rgb_meter.val,
                 l2_recon_loss_rgb_meter.avg,
                 batch_time.avg,
                 batch_time.val,
                 torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                 testing_time]

    write_epoch_to_csv(os.path.join(config.OUTPUT_STATS, f"test_log_table_{test_key}.csv"), curr_list, test_header)


    logger.info(f"Testing takes {datetime.timedelta(seconds=int(testing_time))}")
    logger.info(f"Test Loss: {l2_recon_loss_rgbi_meter.avg:.4f}")

    return l1_recon_loss_rgbi_meter.avg, l2_recon_loss_rgbi_meter.avg, l1_recon_loss_rgb_meter.avg, l2_recon_loss_rgb_meter.avg


if __name__ == '__main__':
    _, config = parse_option()


    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(config.LOCAL_RANK)

    print(f"Process {rank} uses device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    main(config)

