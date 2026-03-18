#import gdown
from pathlib import Path

import os

import lightning
import terratorch.models.necks
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import MLFlowLogger
from mlflow_config_example import *
import albumentations as A
import albumentations.pytorch as ap
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.registry import BACKBONE_REGISTRY
import torch
from terratorch.models import EncoderDecoderFactory
from terratorch.tasks import SemanticSegmentationTask
import matplotlib.pyplot as plt

from terratorch.models.necks import PermuteDims
from models.swin_finetune_tt import build_ft_model
from data.data_ft_tt import initialize_datamodule

import argparse
from config import get_config

import numpy as np
from scipy import interpolate
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from logger import create_logger
import logging
from terratorch.models.decoders import upernet_decoder
import torch.nn.functional as F

import rasterio

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
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
                        help='root of test results folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--train_frac', type=float, default=1.0, help="fraction of training data")

    # distributed training
    parser.add_argument("--local-rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def save_test_to_tif(out_path, sample):
    """
    Saves a reconstructed image in tif format
    Parameters: out_path (string): directory path to save the reconstructed image to
                sample (dict): dictionary with keys 'image', 'mask', 'prediction' and 'filename' that holds all
                                information to save the reconstructed mask with the same georeference as the input image
    """

    filename = os.path.basename(sample["filename"]).split(".")[0]

    # Angenommen, du hast das Eingabebild als TIFF geladen
    with rasterio.open(sample["filename"]) as src:
        # Georeferenzierung des Eingabebildes übernehmen
        transform = src.transform
        crs = src.crs
        dtype = src.dtypes[0]
        width = src.width
        height = src.height


    # Segmentierung als TIFF speichern
    with rasterio.open(os.path.join(out_path, f"pred_{filename}.tif"), "w", driver="GTiff",
                       height=height, width=width, count=1, dtype=dtype,
                       crs=crs, transform=transform) as dst:
        dst.write(sample["prediction"], 1)


def main(config):

    experiment = "tutorial"

    default_root_dir = os.path.join("mlflow_tutorial_experiments", experiment)

    # logger_tt = TensorBoardLogger(save_dir=default_root_dir, name=experiment, log_graph=True)
    # set log_model=True to log the model in the end
    logger_ml = MLFlowLogger(save_dir=default_root_dir, experiment_name=experiment, prefix="",
                             tracking_uri="http://74.63.3.44:5000", log_model=False, synchronous=False) #, run_id='f254ce84935c454f8c9480b8ebfefeed')  # , log_graph=True)

    datamodule = initialize_datamodule(config)

    datamodule.setup("fit")

    val_dataset = datamodule.val_dataset
    train_dataset = datamodule.train_dataset

    print(f"Available samples in the training dataset: {len(train_dataset)}")
    print(f"Input shape of first sample: {train_dataset[0]['image'].shape}")
    print(f"Available samples in the validation dataset: {len(val_dataset)}")
    print(f"Input shape of first sample: {val_dataset[0]['image'].shape}")


    for i in range(0,3):
        train_dataset.plot(train_dataset[i])
        #plt.show()
        fig = plt.gcf()
        logger_ml.experiment.log_figure(run_id=logger_ml.run_id, figure=fig, artifact_file="train_example" + str(i) + ".png")


    model, model_args = build_ft_model(config, logger)

    task = SemanticSegmentationTask(
        model_args=model_args,
        model=model,
        loss="ce",
        lr=2e-5,
        ignore_index=-1,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(default_root_dir,logger_ml.experiment_id,logger_ml.run_id, "checkpoints")
    )

    trainer = Trainer(
        accelerator="auto",  # or specify cpu or gpu
        max_epochs=config.TRAIN.EPOCHS,  # for demo purposes
        default_root_dir=default_root_dir,
        logger=logger_ml,
        callbacks=[
            RichProgressBar(),
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        #log_every_n_steps=config.DATA.BATCH_SIZE,
        enable_model_summary=True,
    )


    if config.MODEL.RESUME != "":
        # Resume training from checkpoint
        trainer.fit(task, ckpt_path=config.MODEL.RESUME, datamodule=datamodule)

    else:
        # Run training
        trainer.fit(model=task, datamodule=datamodule)

    trainer.test(ckpt_path="best", datamodule=datamodule)



    ################## Inference ####################
    # Check the size of the test dataset
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset

    print(f"Available samples in the test dataset: {len(test_dataset)}")
    # Use the trained task for inference and visualization
    task.eval()

    # Get a batch from the test dataloader
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        batch = next(iter(test_loader))
        images = batch["image"].to(task.device)

        outputs = task(images)
        preds = torch.argmax(outputs.output, dim=1).cpu().numpy()

    # Visualize predictions
    for i in range(0, len(preds)): #num_examples):
        sample = {
            "image": batch["image"][i].cpu(),
            "mask": batch["mask"][i],
            "prediction": preds[i],
            "filename": batch["filename"][i],
        }
        save_test_to_tif(config.OUTPUT, sample)
        # if i < 5:
        #     test_dataset.plot(sample)
        #     #plt.show()
        #     fig = plt.gcf()
        #     logger_ml.experiment.log_figure(run_id=logger_ml.run_id, figure=fig, artifact_file="test"+str(i)+".png")


if __name__ == '__main__':
    _, config = parse_option()

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    main(config)
