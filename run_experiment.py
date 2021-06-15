import argparse
import importlib
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import ruamel.yaml as yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import (
    JRDR,
    get_test_loader,
    get_train_transforms,
    get_train_valid_loader,
    get_valid_transforms,
)
from models import DerainCNNModular, FMMRNetModular
from utils import set_seed, validate

AVAILABLE_DATASETS = ["JRDR"]


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def save_experiment_details(logdir, args):
    config_path = os.path.join(logdir, "configs.yaml")
    command_args = dict(defaults=vars(args))
    with open(config_path, "w") as f:
        yaml.dump(command_args, f, default_flow_style=False)

    script_path = os.path.join(logdir, "script.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash")
        f.write("\n")
        f.write("python ")
        f.write(" ".join(sys.argv))


parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="default")
parser.add_argument("--logdir", type=str, default="logs/")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--input_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--dataset", type=str, default="JRDR")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--channel_mul", type=int, default=16)
parser.add_argument("--depth", type=int, default=5)
parser.add_argument("--center_depth", type=int, default=4)
parser.add_argument("--attention_type", type=str, default="channel")
parser.add_argument("--reduction", type=int, default=16)
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--gamma", type=float, default=0.8)
args = parser.parse_args()

device = "cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu"

set_seed(args.seed)
input_size = args.input_size
train_transform = get_train_transforms(input_size)
valid_transform = get_valid_transforms(input_size)
assert args.dataset in AVAILABLE_DATASETS, f"Dataset {args.dataset} not found."
data_class = _import_class(f"dataset.{args.dataset}")
data = data_class(args.data_dir, train_transform=train_transform, valid_transform=valid_transform)

base = FMMRNetModular(
    input_size=args.input_size,
    channel_mul=args.channel_mul,
    depth=args.depth,
    center_depth=args.center_depth,
    attention_type=args.attention_type,
    reduction=args.reduction,
)

model = DerainCNNModular(
    model=base,
    input_size=input_size,
    lr=args.lr,
    gamma=args.gamma,
)

exp_id = os.path.join(args.dataset, args.id)
exp_logdir = os.path.join(args.logdir, exp_id, f"version_{args.seed}")
os.makedirs(exp_logdir, exist_ok=True)
print(f"Logging to {exp_logdir}")

save_experiment_details(exp_logdir, args)

model_checkpoint_dir = os.path.join(exp_logdir, "models")
best_model_file = "{epoch:03d}-{Validation_PSNR:.2f}"
checkpoint_callback = ModelCheckpoint(
    period=1,
    dirpath=model_checkpoint_dir,
    filename=best_model_file,
    verbose=True,
    monitor="Validation_PSNR",
    mode="max",
)

early_stopping = pl.callbacks.EarlyStopping("Validation_PSNR", 0.001, 10, True, "max")
callbacks = [checkpoint_callback, early_stopping]
tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.logdir, name=exp_id, version=args.seed, log_graph=True)

epochs = args.epochs
trainer = Trainer(
    gpus=1 if device == "cuda" else 0,
    callbacks=callbacks,
    min_epochs=epochs,
    max_epochs=epochs + 5,
    progress_bar_refresh_rate=20,
    weights_summary="top",
    benchmark=True,
    logger=tb_logger,
)

trainer.fit(model, datamodule=data)

final_checkpoint_file = os.path.join(model_checkpoint_dir, "final_epoch.pth")
torch.save(model.state_dict(), final_checkpoint_file)

model.eval()
model = model.to(device)
validate(model, data.train_loader, "train", exp_logdir)
validate(model, data.val_loader, "valid", exp_logdir)
validate(model, data.test_loader, "test", exp_logdir)
