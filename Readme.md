# GFM-aerial

Adaptation of the [GFM](https://github.com/mmendiet/GFM) by Mendieta et al. (2023) to German aerial images in RGBI format. 

## Model structure
<img src="img/Model_structure.png"/>

## Getting started

### Dependencies
So far, this code has only been tested on Linux systems with a single GPU.
CUDA compatibility and a GPU are necessary.


### Installing
In a terminal, go to the environment folder and run:
```
git clone https://github.com/LUP-LuftbildUmweltPlanung/GFM_aerial
cd GFM_aerial/environment/
conda env create -f environment.yml 
conda activate gfm-aerial
cd ../GFM-main/git_reps/
git clone https://github.com/microsoft/SimMIM
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Logging with MLflow
If you want to log your statistics with MLflow, you have to modify GFM-main/mlflow_config_example.py with your own configurations. 


### Executing pretraining - GFM teacher
If you want to use the GFM by Mendieta et al. as teacher, download it from [here](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBa1RuNzZtOTA3T1RocFJKakg4ZWhmc2tiZ0NMWHc%5FZT1aSnJlRm8&id=93B3D3BDA9EFE744%21100937&cid=93B3D3BDA9EFE744)
You need to set the following values in gfm_aerial_pretraining.yaml:
```
TEACHER_WINDOW_SIZE: 6
TEACHER_IMG_SIZE: 192
```

Modify the yaml file with the paths to your datasets.

Run pretraining with this line:
```
python -m torch.distributed.launch --nproc_per_node 1 GFM-main/main_teacher.py --cfg GFM-main/configs/gfm_aerial_pretraining.yaml --batch-size 32 --tag gfm --pretrained GFM-main/output/simmim_finetune/gfm.pth
```

### Executing pretraining - Swin teacher
If you want to train with the Swin-transformer as teacher, execute this code:
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
```
You need to set the following values in gfm_aerial_pretraining.yaml:
```
TEACHER_WINDOW_SIZE: 7
TEACHER_IMG_SIZE: 224
```

Modify the yaml file with the paths to your datasets.

Run pretraining with this line:
```
python -m torch.distributed.launch --nproc_per_node 1 GFM-main/main_teacher.py --cfg GFM-main/configs/gfm_aerial_pretraining.yaml --batch-size 32 --tag swin --pretrained GFM-main/output/simmim_finetune/swin_base_patch4_window7_224_22k.pth
```

### Testing
Make sure to use the same variables of teacher and student that were used during training.
If you want to test the original GFM with Swin teacher, set the variables according to Swin-teacher pretraining and change:
```
IN_CHANS: 3
```

Test the model with this line:
```
python -m torch.distributed.launch --nproc_per_node 1 GFM-main/main_testing.py --cfg GFM-main/configs/gfm_aerial_pretraining.yaml --batch-size 32 --tag gfm_aerial_testing --pretrained GFM-main/output/simmim_finetune/gfm.pth --resume PATH_TO_BEST_CHECKPOINT 
```
If you want to save the reconstructed images in an lmdb file, add a path to the OUTPUT_LMDB variable of your yaml file.

## Authors

Vera Sons

## License

This project is licensed under the GNU General Public Licence, Version 3 (GPLv3) License - see the [LICENSE](LICENSE) file for details

## References

This work builds upon the paper [Towards Geospatial Foundation Models via Continual Pretraining](https://arxiv.org/abs/2302.04476)
and [code](https://github.com/mmendiet/GFM) by Mendieta et al.
