import os
import torch
from typing import Callable

from torchgeo.datasets.geo import NonGeoClassificationDataset

import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp

from data.data_simmim import ensure_four_channels_tensor

import rasterio



class RGBI_dataset(NonGeoClassificationDataset):
    """
    Dataset features:

    * four spectral bands - RGBI
    """

    base_dir = os.path.join("4_bands", "Images")

    splits = ["train", "val", "test"]

    def __init__(self, classes, root="data", split="train", transform=None):
        """Initialize a new RGBI dataset instance.

        Args:
            classes: list of classes in the dataset
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transform: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            RuntimeError: if data is not found
        """
        assert split in self.splits
        self.classes = classes
        self.root = root
        self.transforms = transform
        self._verify()

        self.samples = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # ToDo:adapt for subset of all classes, make one-hot-encoded

        valid_fns = set()
        with open(os.path.join(self.root, f"{split}.txt")) as f:
            for fn in f:
                fn_strip = fn.strip()
                if os.path.exists(fn_strip):
                    class_name = os.path.basename(os.path.dirname(fn_strip))
                    self.samples.append((fn_strip, self.class_to_idx[class_name]))
                    valid_fns.add(os.path.basename(fn_strip))

        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transform,
            is_valid_file=is_in_split,
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset.
        Raises:
            RuntimeError: if dataset is missing
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.base_dir)
        if os.path.exists(filepath):
            return
        else:
            raise RuntimeError(
                "Dataset not found in `root` directory"
            )


    def __getitem__(self, index: int):
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        path, label = self.samples[index]
        with rasterio.open(path) as src:
            image = src.read()

        image = torch.from_numpy(image).float()

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.imgs)


def build_transform(config, split='train'):
    """Return a transform function that can be applied on the input image.
    Args:
        config: configuration parameters such as image size or interpolation
        split: train, vali or test
    Returns:
        transform function"""
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0 or config.AUG.CUTMIX_MINMAX is not None
    if split == 'train' and mixup_active:
        transforms = T.Compose([
            T.Lambda(lambda img: ensure_four_channels_tensor(img)),
            T.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE), interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(lambda img: img / 255.0 if img.max() > 1 else img), #otherwise done with ToTensor()
            T.Normalize(mean=torch.tensor(list(IMAGENET_DEFAULT_MEAN) + [0.5947974324226379]),
                        std=torch.tensor(list(IMAGENET_DEFAULT_STD) + [0.19213160872459412])),
        ])
    elif split == 'train':
        transforms = T.Compose([
            T.Lambda(lambda img: ensure_four_channels_tensor(img)),
            T.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE), interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Lambda(lambda img: img / 255.0 if img.max() > 1 else img), #otherwise done with ToTensor()
            T.Normalize(mean=torch.tensor(list(IMAGENET_DEFAULT_MEAN) + [0.5947974324226379]),
                        std=torch.tensor(list(IMAGENET_DEFAULT_STD) + [0.19213160872459412])),
        ])
    else:
        transforms = T.Compose([
            T.Lambda(lambda img: ensure_four_channels_tensor(img)),
            T.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE), interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
            T.Lambda(lambda img: img / 255.0 if img.max() > 1 else img), #otherwise done with ToTensor()
            T.Normalize(mean=torch.tensor(list(IMAGENET_DEFAULT_MEAN) + [0.5947974324226379]),
                        std=torch.tensor(list(IMAGENET_DEFAULT_STD) + [0.19213160872459412])),
        ])
    return transforms