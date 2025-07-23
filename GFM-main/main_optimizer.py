# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import os
import gc
import numpy as np

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.teacher import build_simmim
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger

from main_teacher import train_one_epoch, validate_one_epoch, parse_option



# Optimize PyTorch precision
torch.set_float32_matmul_precision('medium')

class HyperOpti:
    def __init__(self, config, logger):
        config.defrost()
        self.external_config = config
        self.logger = logger

    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        drop_rate = Float("drop_rate", (0.0, 1.0), default=0.0)
        batch_size = Integer("batch_size", (60, 200), default=128)
        weight_decay = Float("weight_decay", (0.0, 1.0), default=0.05)
        base_lr = Float("base_lr", (0.00001, 1.0), default=2e-4, log=True)

        # Add all hyperparameters at once:
        cs.add([drop_rate, batch_size, weight_decay, base_lr])

        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        data_loader_train = build_loader(self.external_config, self.logger, is_pretrain=True, is_train=True)
        data_loader_vali_temp_ind = build_loader(self.external_config, self.logger, is_pretrain=True, is_train=False, vali_key=0)
        data_loader_vali_spa_ind = build_loader(self.external_config, self.logger, is_pretrain=True, is_train=False, vali_key=1)
        data_loader_vali_temp_spa_ind = build_loader(self.external_config, self.logger, is_pretrain=True, is_train=False, vali_key=2)

        self.external_config.defrost()
        self.external_config.AMP_OPT_LEVEL= "O0"
        self.external_config.MODEL.DROP_RATE = config["drop_rate"]
        self.external_config.DATA.BATCH_SIZE = config["batch_size"]
        self.external_config.TRAIN.WEIGHT_DECAY = config["weight_decay"]
        self.external_config.TRAIN.BASE_LR = config["base_lr"]

        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = self.external_config.TRAIN.BASE_LR * self.external_config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = self.external_config.TRAIN.WARMUP_LR * self.external_config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = self.external_config.TRAIN.MIN_LR * self.external_config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        # gradient accumulation also need to scale the learning rate
        if self.external_config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * self.external_config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * self.external_config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * self.external_config.TRAIN.ACCUMULATION_STEPS
        self.external_config.TRAIN.BASE_LR = linear_scaled_lr
        self.external_config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        self.external_config.TRAIN.MIN_LR = linear_scaled_min_lr
        self.external_config.freeze()

        model = build_simmim(self.external_config, logger)
        model.cuda()

        optimizer = build_optimizer(self.external_config, model, logger, is_pretrain=True)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.external_config.LOCAL_RANK], broadcast_buffers=False)

        lr_scheduler = build_scheduler(self.external_config, optimizer, len(data_loader_train))

        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        for epoch in range(0, int(np.ceil(budget))):
            data_loader_train.sampler.set_epoch(epoch)

            train_loss = train_one_epoch(self.external_config, model, data_loader_train, optimizer, epoch, lr_scheduler, logger)
        val_loss_temp_ind = validate_one_epoch(self.external_config, model, data_loader_vali_temp_ind, epoch, logger, val_key="temp_ind")
        val_loss_spa_ind = validate_one_epoch(self.external_config, model, data_loader_vali_spa_ind, epoch, logger, val_key="spa_ind")
        val_loss_temp_spa_ind = validate_one_epoch(self.external_config, model, data_loader_vali_temp_spa_ind, epoch, logger, val_key="temp_spa_ind")
        avg_val_loss = (val_loss_temp_ind + val_loss_spa_ind + val_loss_temp_spa_ind)/3

        del model
        del lr_scheduler
        del data_loader_train
        del data_loader_vali_temp_ind
        del data_loader_vali_spa_ind
        del data_loader_vali_temp_spa_ind 
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        # TODO: which loss?
        
        return avg_val_loss
    
    def optimize(self):
        pass


if __name__ == '__main__':
    _, config = parse_option()


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
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=1, name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    hyperopti = HyperOpti(config, logger)

    facades = []
    for intensifier_object in [SuccessiveHalving, Hyperband]:
        # Define our environment variables
        scenario = Scenario(
            hyperopti.configspace,
            walltime_limit=30,  # After 60 seconds, we stop the hyperparameter optimization
            n_trials=5,  # Evaluate max 500 different trials
            min_budget=1,  # Train the NN using a hyperparameter configuration for at least 1 epoch
            max_budget=5,  # Train the NN using a hyperparameter configuration for at most 25 epochs
            n_workers=1,
        )

        # We want to run five random configurations before starting the optimization.
        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

        # Create our intensifier
        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

        # Create our SMAC object and pass the scenario and the train method
        smac = MFFacade(
            scenario,
            hyperopti.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

        # Let's optimize
        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(hyperopti.configspace.get_default_configuration())
        print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")

        facades.append(smac)


