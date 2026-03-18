from pathlib import Path
import albumentations as A
import albumentations.pytorch as ap
from terratorch.datamodules import GenericNonGeoSegmentationDataModule


def initialize_datamodule(config):
    """ Initialize the datamodule for semantic segmentation. The datamodule consists of train, validation and test subsets."""

    dataset_path = Path(config.DATA.DATA_TRAIN_PATH)

    print(dataset_path)

    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=0,

        # We use the same roots for train/val/test and select samples via the given split files
        train_data_root=dataset_path / "trai/img_tiles",
        train_label_data_root=dataset_path / "trai/mask_tiles",
        val_data_root=dataset_path / "vali/img_tiles",
        val_label_data_root=dataset_path / "vali/mask_tiles",
        test_data_root=dataset_path / "test/img_tiles",
        test_label_data_root=dataset_path / "test/mask_tiles",

        # Split files
        train_split=dataset_path / "splits/train_data",
        val_split=dataset_path / "splits/vali_data",
        test_split=dataset_path / "splits/test_data",

        # File patterns inside the roots above
        img_grep="*.tif",
        label_grep="*.tif",

        # Data transforms
        train_transform=[
            A.RandomCrop(width=config.DATA.TEACHER_IMG_SIZE, height=config.DATA.TEACHER_IMG_SIZE),
            A.D4(),  # random flips and rotations to stabilize training
            ap.ToTensorV2(),
        ],
        val_transform=[A.RandomCrop(width=config.DATA.TEACHER_IMG_SIZE, height=config.DATA.TEACHER_IMG_SIZE),  ap.ToTensorV2()],
        test_transform=[A.RandomCrop(width=config.DATA.TEACHER_IMG_SIZE, height=config.DATA.TEACHER_IMG_SIZE),  ap.ToTensorV2()],

        dataset_bands=[0, 1, 2, 3],
        output_bands=[0, 1, 2, 3],

        # RGB visualization uses channels [R,G,B] = [3,2,1]
        rgb_indices=[0, 1, 2],
        num_classes=len(config.DATA.CLASSES),

        means=[0.485, 0.456, 0.406, 0.5947974324226379],
        stds=[0.229, 0.224, 0.225, 0.19213160872459412],

        no_data_replace=0,
        no_label_replace=-1
    )
    return datamodule