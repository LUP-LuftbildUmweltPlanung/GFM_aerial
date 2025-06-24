# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import warnings

import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

######
import data.ben_data as bend
import os

from safetensors.numpy import load
import lmdb

from typing import TypeVar, Optional, Iterator
######


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    def __init__(self, config, is_train=True, vali_key=None):

        if is_train:
            data_path = config.DATA.DATA_TRAIN_PATH

            if 'bigearthnet' in data_path:
                self.transform_img = bend.build_transform(config, split='train')
            elif data_path.endswith(".lmdb"):
                self.transform_img = T.Compose([
                    T.Lambda(lambda img: self.ensure_four_channels_tensor(img)),
                    T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                    T.RandomHorizontalFlip(),
                    T.Lambda(lambda img: img / 255.0 if img.max() > 1 else img), #otherwise done with ToTensor()
                    T.Normalize(mean=torch.tensor(list(IMAGENET_DEFAULT_MEAN) + [0.5947974324226379]),
                                std=torch.tensor(list(IMAGENET_DEFAULT_STD) + [0.19213160872459412])),
                ])
            elif 'GeoPileV0' in data_path and not data_path.endswith(".lmdb"):
                self.transform_img = T.Compose([
                    T.Lambda(lambda img: img.convert('RGBA') if img.mode != 'RGBA' else img),
                    T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(list(IMAGENET_DEFAULT_MEAN) + [0.5947974324226379]), std=torch.tensor(list(IMAGENET_DEFAULT_STD) + [0.19213160872459412])),
                ])
            else:
                raise NotImplementedError
        else:
            data_path = config.DATA.DATA_VALI_PATH[vali_key]

            if data_path.endswith(".lmdb"):
                self.transform_img = T.Compose([
                    T.Lambda(lambda img: self.ensure_four_channels_tensor(img)),
                    T.Lambda(lambda img: img / 255.0 if img.max() > 1 else img), #otherwise done with ToTensor()
                    T.Normalize(mean=torch.tensor(list(IMAGENET_DEFAULT_MEAN) + [0.5947974324226379]),
                                std=torch.tensor(list(IMAGENET_DEFAULT_STD) + [0.19213160872459412])),
                ])
            else:
                raise NotImplementedError
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def ensure_four_channels_tensor(self, img):
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:  # If there are only 3 channels (C, H, W)
                alpha_channel = torch.full((1, img.shape[1], img.shape[2]), 0.5, dtype=img.dtype, device=img.device)
                img = torch.cat([img, alpha_channel], dim=0)  # Add the fourth channel
        else:
            raise NotImplementedError
        return img

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


class LMDBSafetensorDataset(Dataset):
    """
    LMDB Dataset for loading data from lmdb instead of tiff files.
    """

    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)

        with self.env.begin() as txn:
            self.keys = [key.decode() for key, _ in txn.cursor()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        with self.env.begin() as txn:
            safetensor_data = txn.get(key.encode())

        if safetensor_data is None:
            raise KeyError(f"No entry for '{key}' was found in the lmdb!")

        bands_dict = load(safetensor_data)

        self.selected_bands = sorted(list(bands_dict.keys()))

        band_arrays = [bands_dict[b] for b in self.selected_bands if b in bands_dict]

        if len(band_arrays) == 0:
            raise ValueError(f"No band entries found for '{key}'!")

        stacked_array = np.stack(band_arrays)  # Shape: (C, H, W)

        tensor = torch.from_numpy(stacked_array).float()

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, key

def build_loader_simmim(config, logger, is_train=True, vali_key=None):
    if is_train:
        data_path = config.DATA.DATA_TRAIN_PATH
        logger.info(f"Train data path: {data_path}")
    else:
        data_path = config.DATA.DATA_VALI_PATH[vali_key]
        logger.info(f"Vali data path: {data_path}")

    transform = SimMIMTransform(config, is_train, vali_key)
    logger.info('Pre-train data transform:\n{}'.format(transform.transform_img))

    if data_path.endswith(".lmdb"):
        logger.info(f"Load LMDB: {data_path}")
        dataset = LMDBSafetensorDataset(data_path, transform)
    elif 'GeoPileV0' in data_path and not data_path.endswith(".lmdb"):
        datasets = []
        for ds in os.listdir(data_path):
            datasets.append(ImageFolder(os.path.join(data_path, ds), transform))
        dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        dataset = ImageFolder(data_path, transform)

    logger.info(f'Build dataset: images = {len(dataset)}')

    # Sampler: Train = shuffle, Val = no shuffle
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=is_train
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=is_train,  # just for training
        collate_fn=collate_fn
    )

    return dataloader

class my_sampler(DistributedSampler):
    def __init__(self, dataset, num_replicas = None, batch_size=64,
                 rank = None, shuffle = True, seed = 0, drop_last = False, partitions=[]):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.partitions = partitions
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            # print('here')
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_list = []
            start = 0
            for p in self.partitions:
                # indices = torch.randperm(len(self.dataset), generator=g).tolist()
                indices = torch.randperm(p, generator=g).tolist()
                indices = [i+start for i in indices]
                batches = zip(*(iter(indices),) * (self.batch_size*self.num_replicas))
                batch_list.extend(batches)
                start += p
            random.shuffle(batch_list)
            indices = [item for sublist in batch_list for item in sublist]
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)