# import relevant libraries
import os

import torch
import wandb

from CrossModalGraph.configs.config import get_cfg
from CrossModalGraph.model.meta_arch import CNN, LSTM, GraphNN

# from CrossModalGraph.model.meta_arch import CNN, LSTM, GraphNN
from CrossModalGraph.train_utils.train_loop import DefaultTrainer
from CrossModalGraph.utils.logger import setup_logger

setup_logger()


device = "cuda" if torch.cuda.is_available() else "cpu"

# set default configs
cfg = get_cfg()
cfg.merge_from_file("configs/AudioSet.yaml")
cfg.DEVICE = device
cfg.MODEL.DEVICE = device
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"  # "RepeatFactorTrainingSampler"
# specify the disered classes
cfg.DATALOADER.DISERED_CLASSES = ["Speech", "Music"]
# cfg.DATALOADER.DISERED_CLASSES = [
#     "Speech",
#     "Music",
#     "Cat",
#     "Dog",
#     "Car",
#     "Train",
#     "Bird",
#     "Gunshot, gunfire",
# ]

# cfg.DATALOADER.DISERED_CLASSES = [
#     "Aircraft",
#     "Ambulance (siren)",
#     "Bicycle",
#     "Bird",
#     "Boom",
#     "Bus",
#     "Camera",
#     "Car",
#     "Cash register",
#     "Cat",
#     "Cattle, bovinae",
#     "Church bell",
#     "Clock",
#     "Dog",
#     "Mechanical fan",
#     "Fireworks",
#     "Goat",
#     "Gunshot, gunfire",
#     "Hammer",
#     "Horse",
#     "Motorcycle",
#     "Ocean",
#     "Pant",
#     "Pig",
#     "Printer",
#     "Rain",
#     "Sawing",
#     "Sewing machine",
#     "Skateboard",
#     "Stream",
#     "Thunderstorm",
#     "Train",
#     "Truck",
# ]
cfg.MODEL.OUT_DIM = len(cfg.DATALOADER.DISERED_CLASSES)
cfg.WANDB.EXP_NAME = f"{cfg.SOLVER.ITERS_TO_ACCUMULATE}_grd_accu"
cfg.WANDB.GP_NAME = "without_graph_model"
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Call wandb.finish() to upload your TensorBoard logs to W&B
wandb.finish()
