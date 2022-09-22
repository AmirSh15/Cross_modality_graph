# import relevant libraries
import os

import torch
import wandb

from CrossModalGraph.model.meta_arch import GraphNN

from CrossModalGraph.train_utils.train_loop import DefaultTrainer
from CrossModalGraph.utils.logger import setup_logger
from CrossModalGraph.utils.utils import setup_random_seed, get_random_string, read_config

setup_logger()

if __name__ == "__main__":

    # set default configs
    cfg = read_config()

    # set random seed
    setup_random_seed(cfg.SEED)

    # set wandb config
    cfg.MODEL.OUT_DIM = len(cfg.DATALOADER.DISERED_CLASSES)
    cfg.WANDB.EXP_NAME = f"{cfg.SOLVER.ITERS_TO_ACCUMULATE}_grd_accu"
    cfg.WANDB.GP_NAME = "without_graph_model_with_grd_clip_without_ft"
    cfg.WANDB.ID = get_random_string(len(cfg.WANDB.EXP_NAME)+cfg.SOLVER.ITERS_TO_ACCUMULATE)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Call wandb.finish() to upload your TensorBoard logs to W&B
    wandb.finish()
