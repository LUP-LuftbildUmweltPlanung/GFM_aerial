from pathlib import Path
import albumentations as A
import albumentations.pytorch as ap
from terratorch.datamodules import GenericNonGeoSegmentationDataModule


def initialize_datamodule(config):
    """ Initialize the datamodule for semantic segmentation. The datamodule consists of train, validation and test subsets."""

    dataset_path = Path(config.DATA.DATA_TRAIN_PATH)

    #print(dataset_path)

    if config.MODEL.SWIN.IN_CHANS == 4:
        set_means = [0.485, 0.456, 0.406, 0.5947974324226379]
        set_stds = [0.229, 0.224, 0.225, 0.19213160872459412]
        data_bands = [0, 1, 2, 3]
    elif config.MODEL.SWIN.IN_CHANS == 5:
        set_means = [0.485, 0.456, 0.406, 0.5947974324226379, 0.5]
        set_stds = [0.229, 0.224, 0.225, 0.19213160872459412, 0.5]
        data_bands = [0, 1, 2, 3, 4]

    if config.DATA.DATA_FORMAT == "uint16":
        scale_factor = 65535.0
    else: # uint8 expected
        scale_factor = 255.0

    #rescale_image_transform = A.Lambda(name="rescale_image", image=rescale_image(img=image, scale_factor=scale_factor))

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
            A.RandomCrop(width=config.DATA.IMG_SIZE, height=config.DATA.IMG_SIZE),
            A.D4(),  # random flips and rotations to stabilize training
            A.Normalize(mean=set_means, std=set_stds, max_pixel_value=scale_factor, normalization='standard'),
            ap.ToTensorV2(),
            #A.Lambda(name="rescale_image", image=rescale_image),

        ],
        val_transform=[A.RandomCrop(width=config.DATA.IMG_SIZE, height=config.DATA.IMG_SIZE),
                       A.Normalize(mean=set_means, std=set_stds, max_pixel_value=scale_factor,
                                   normalization='standard'),
                       ap.ToTensorV2(),
                       ],
        test_transform=[A.RandomCrop(width=config.DATA.IMG_SIZE, height=config.DATA.IMG_SIZE),
                        A.Normalize(mean=set_means, std=set_stds, max_pixel_value=scale_factor,
                                    normalization='standard'),
                        ap.ToTensorV2(),
                        ],

        dataset_bands=data_bands,
        output_bands=data_bands,

        # RGB visualization uses channels [R,G,B] = [3,2,1]
        rgb_indices=[0, 1, 2],
        num_classes=len(config.DATA.CLASSES),

        means=set_means,
        stds=set_stds,

        no_data_replace=0,
        no_label_replace=-1
    )
    return datamodule