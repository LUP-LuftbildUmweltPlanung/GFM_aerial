# Project Title

Adaptation of the [GFM](https://github.com/mmendiet/GFM) by Mendieta et al. for German aerial images in RGBI format. 

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies
So far, this code has only been tested on Linux.
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

### Executing pretraining
If you want to use the GFM by Mendieta et al. as teacher, download it from [here](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBa1RuNzZtOTA3T1RocFJKakg4ZWhmc2tiZ0NMWHc%5FZT1aSnJlRm8&id=93B3D3BDA9EFE744%21100937&cid=93B3D3BDA9EFE744)
You need to set the following values in simmim_pretrain__swin_base__img192_window6__100ep.yaml:
```
TEACHER_WINDOW_SIZE: 6
TEACHER_IMG_SIZE: 192
```

If you want to train with the Swin-transformer as teacher, execute this code:
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
```
You need to set the following values in simmim_pretrain__swin_base__img192_window6__100ep.yaml:
```
TEACHER_WINDOW_SIZE: 7
TEACHER_IMG_SIZE: 224
```

Modify the yaml file with the paths to your datasets.

Run pretraining with this line:
```
python -m torch.distributed.launch --nproc_per_node 1 main_teacher.py --cfg configs/simmim_pretrain__swin_base__img192_window6__100ep.yaml --batch-size 8 --tag gfm --pretrained output/simmim_finetune/gfm.pth
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names / contact info / GitHub Acc


## Known Issues

* list known issues or limitations


## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## References

Inspiration, code snippets, etc.
