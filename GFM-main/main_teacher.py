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

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow_config import *

from config import get_config
# from models import build_model
from models.teacher import build_simmim
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None



# Optimize PyTorch precision
torch.set_float32_matmul_precision('medium')

# Set MLflow request timeout
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "3500"
print(f" MLflow Tracking URI Set: {mlflow.get_tracking_uri()}")

# Initialize MLflow Client
client = MlflowClient()

# Define Experiment Name
experiment_name = "GFMaerial_test_half_train"

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
    data_loader_train = build_loader(config, logger, is_pretrain=True, is_train=True)
    data_loader_vali_temp_ind = build_loader(config, logger, is_pretrain=True, is_train=False, vali_key=0)
    data_loader_vali_spa_ind = build_loader(config, logger, is_pretrain=True, is_train=False, vali_key=1)
    data_loader_vali_temp_spa_ind = build_loader(config, logger, is_pretrain=True, is_train=False, vali_key=2)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_simmim(config, logger)
    model.cuda()
    logger.info(str(model))

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


    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    best_val_loss = float('inf')
    loss_log_table = []
    val_temp_ind_table = []
    val_spa_ind_table = []
    val_temp_spa_ind_table = []
    train_log_table = []
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, train_log_table)
        val_loss_temp_ind = validate_one_epoch(config, model, data_loader_vali_temp_ind, epoch, val_temp_ind_table)
        val_loss_spa_ind = validate_one_epoch(config, model, data_loader_vali_spa_ind, epoch, val_spa_ind_table)
        val_loss_temp_spa_ind = validate_one_epoch(config, model, data_loader_vali_temp_spa_ind, epoch, val_temp_spa_ind_table)
        avg_val_loss = (val_loss_temp_ind + val_loss_spa_ind +val_loss_temp_spa_ind)/3

        if dist.get_rank() == 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(config, epoch, model_without_ddp, best_val_loss, optimizer, lr_scheduler, logger)

        if dist.get_rank() == 0:
            # Mlflow
            mlflow.log_metric("val_loss_avg", avg_val_loss, step=epoch) # Average of all three validation losses
            mlflow.log_metric("val_loss_temp", val_loss_temp_ind, step=epoch) # Validation loss on temporally independent samples
            mlflow.log_metric("val_loss_spa", val_loss_spa_ind, step=epoch) # Validation loss on spatially independent samples
            mlflow.log_metric("val_loss_temp_spa", val_loss_temp_spa_ind, step=epoch) # Validation loss on spatially AND temporally independent samples
            mlflow.log_metric("train_loss", train_loss, step=epoch) # Train loss

            loss_log_table.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss_avg": avg_val_loss,
                "val_loss_temp": val_loss_temp_ind,
                "val_loss_spa": val_loss_spa_ind,
                "val_loss_temp_spa": val_loss_temp_spa_ind
            })

        #if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        #    save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if dist.get_rank() == 0:
        import pandas as pd
        loss_log_df = pd.DataFrame(loss_log_table)
        mlflow.log_table(loss_log_df, artifact_file="losses_per_epoch.json")

        val_temp_ind_log_df = pd.DataFrame(val_temp_ind_table)
        mlflow.log_table(val_temp_ind_log_df, artifact_file="val_temp_ind_log.json")
        val_spa_ind_log_df = pd.DataFrame(val_spa_ind_table)
        mlflow.log_table(val_spa_ind_log_df, artifact_file="val_spa_ind_log.json")
        val_temp_spa_ind_log_df = pd.DataFrame(val_temp_spa_ind_table)
        mlflow.log_table(val_temp_spa_ind_log_df, artifact_file="val_temp_spa_ind_log.json")


        train_log_df = pd.DataFrame(train_log_table)
        mlflow.log_table(train_log_df, artifact_file="train_log.json")

    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, train_log_table):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    dist_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    curr_dict = {}
    #for idx, (img, mask, _) in enumerate(data_loader):
    #for idx, (img, mask) in enumerate(data_loader):
    for idx, batch in enumerate(data_loader):
        #print(batch[0])
        #print(f"type batch: {type(batch)}")
        #print(type(data_loader.dataset))
        #exit()
        #break
        if config.DATA.DATA_TRAIN_PATH.endswith(".lmdb"):
            img, mask = batch  # Falls LMDB nur 2 Werte liefert
        else:
            img, mask, _ = batch  # Falls GeoPileV0 ein drittes Element zurückgibt

        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        loss, reconstruction_loss, distillation_loss = model(img, mask)

        #logger.info(f"losses: {loss}, {reconstruction_loss}, {distillation_loss}")

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            reconstruction_loss = reconstruction_loss / config.TRAIN.ACCUMULATION_STEPS
            distillation_loss = distillation_loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        recon_loss_meter.update(reconstruction_loss.item(), img.size(0))
        dist_loss_meter.update(distillation_loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            curr_dict = {
                "epoch": epoch,
                "loss_avg": loss_meter.avg,
                "loss_value": loss_meter.val,
                "reconstruction_loss_avg": recon_loss_meter.avg,
                "reconstruction_loss_val": recon_loss_meter.val,
                "distillation_loss_avg": dist_loss_meter.avg,
                "distillation_loss_val": dist_loss_meter.val,
                "batch_time_avg": batch_time.avg,
                "batch_time_val": batch_time.val,
                "memory_used": memory_used,
                "eta": datetime.timedelta(seconds=int(etas)),
                "grad_norm_avg": float(norm_meter.avg),
                "grad_norm_val": float(norm_meter.val)
            }

            #logger.info(type(norm_meter.avg),  type(norm_meter.val))

    epoch_time = time.time() - start

    curr_dict["epoch_time"] = epoch_time
    train_log_table.append(curr_dict)

    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg


@torch.no_grad()
def validate_one_epoch(config, model, data_loader, epoch, val_log_table, lmdb_key=True):


    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    dist_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    num_steps = len(data_loader)

    curr_dict = {}

    for idx, batch in enumerate(data_loader):
        if lmdb_key:
            img, mask = batch
        else:
            img, mask, _ = batch

        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        loss, reconstruction_loss, distillation_loss  = model(img, mask)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        recon_loss_meter.update(reconstruction_loss.item(), img.size(0))
        dist_loss_meter.update(distillation_loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Val:   [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            curr_dict = {
                "epoch": epoch,
                "loss_avg": loss_meter.avg,
                "loss_value": loss_meter.val,
                "reconstruction_loss_avg": recon_loss_meter.avg,
                "reconstruction_loss_val": recon_loss_meter.val,
                "distillation_loss_avg": dist_loss_meter.avg,
                "distillation_loss_val": dist_loss_meter.val,
                "batch_time_avg": batch_time.avg,
                "batch_time_val": batch_time.val,
                "memory_used": memory_used,
                "eta": datetime.timedelta(seconds=int(etas))
            }

    epoch_time = time.time() - start

    curr_dict["epoch_time"] = epoch_time
    val_log_table.append(curr_dict)


    logger.info(f"EPOCH {epoch} validation takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"Validation Loss: {loss_meter.avg:.4f}")

    return loss_meter.avg


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
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    #logger.info(config.dump())

    #logger.info("Train dataset: ", config.DATA.DATA_TRAIN_PATH)


    # -------------------- MLflow Run Start --------------------
    if dist.get_rank() == 0:
        with mlflow.start_run(run_name=config.MODEL.NAME):
            # Add tags for traceability
            mlflow.set_tag("mlflow.user", socket.gethostname())  # Will show as "Created by"
            mlflow.set_tag("mlflow.source.name", os.path.abspath(__file__))  # Will show as "Source"
            #mlflow.set_tag("mlflow.dataset", config.DATA.DATA_TRAIN_PATH)
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
            main(config)
    else:
        main(config)

