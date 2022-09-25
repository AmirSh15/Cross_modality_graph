# import relevant libraries
import os

import torch

import wandb
from CrossModalGraph.model.meta_arch import GraphNN
from CrossModalGraph.train_utils.train_loop import DefaultTrainer
from CrossModalGraph.utils.logger import setup_logger
from CrossModalGraph.utils.utils import (get_random_string, read_config,
                                         setup_random_seed)

setup_logger()

if __name__ == "__main__":

    # set default configs
    cfg = read_config()

    # set random seed
    setup_random_seed(cfg.SEED)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Call wandb.finish() to upload your TensorBoard logs to W&B
    wandb.finish()
