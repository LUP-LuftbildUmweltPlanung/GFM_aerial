from .data_simmim import build_loader_simmim, build_loader_vali
from .data_finetune import build_loader_finetune

def build_loader(config, logger, is_pretrain, is_train=True, vali_key=None):
    if is_pretrain:
        return build_loader_simmim(config, logger, is_train, vali_key)
    else:
        return build_loader_finetune(config, logger)