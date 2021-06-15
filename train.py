import argparse
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
from models import DerainCNN, DerainCNNModular
from utils import set_seed, validate

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="default")
parser.add_argument("--logdir", type=str, default="logs/")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--input_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
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
if args.dataset == "JRDR":
    dataset_dir = os.path.join("data", "JRDR")
    test_dir = os.path.join("data", "JRDR")
    train_data = JRDR(root=dataset_dir, transform=train_transform)
    valid_data = JRDR(root=dataset_dir, transform=valid_transform)
    test_data = JRDR(root=test_dir, split="test", transform=valid_transform)
else:
    raise NotImplementedError(args.dataset)

train_loader, valid_loader = get_train_valid_loader(train_data, valid_data, show_sample=False)
test_loader = get_test_loader(test_data)

model = DerainCNNModular(
    input_size=input_size,
    channel_mul=args.channel_mul,
    depth=args.depth,
    center_depth=args.center_depth,
    attention_type=args.attention_type,
    reduction=args.reduction,
    lr=args.lr,
    gamma=args.gamma,
)

exp_id = os.path.join(args.dataset, args.id)
logdir = args.logdir

exp_log_dir = os.path.join(logdir, exp_id, str(args.seed))
os.makedirs(exp_log_dir, exist_ok=True)
print(f"Logging to {exp_log_dir}")
print(f"Checkpoint saved to {exp_log_dir}")

config_path = os.path.join(exp_log_dir, "configs.yaml")
command_args = dict(defaults=vars(args))
with open(config_path, "w") as f:
    yaml.dump(command_args, f, default_flow_style=False)

script_path = os.path.join(exp_log_dir, "script.sh")
with open(script_path, "w") as f:
    f.write("#!/bin/bash")
    f.write("\n")
    f.write("python ")
    f.write(" ".join(sys.argv))

checkpoint_file = os.path.join(exp_log_dir, "latest_epoch.pth")
best_model_dir = os.path.join(exp_log_dir, "models")
best_model_file = "{valid/PSNR:.2f}"
report_file = os.path.join(exp_log_dir, "report.txt")


class SaveModelCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        torch.save(pl_module.state_dict(), checkpoint_file)
        validate(pl_module, train_loader, "train", exp_log_dir)


checkpoint_callback = ModelCheckpoint(
    period=1,
    dirpath=best_model_dir,
    filename=best_model_file,
    verbose=True,
    monitor="valid/PSNR",
    mode="max",
)

early_stopping = pl.callbacks.EarlyStopping("valid/PSNR", 0.001, 10, True, "max")
profiler = pl.profiler.AdvancedProfiler(report_file)
callbacks = [SaveModelCallback(), checkpoint_callback, early_stopping]
tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name=exp_id, version=args.seed, log_graph=True)

epochs = args.epochs
trainer = Trainer(
    gpus=1 if device == "cuda" else 0,
    profiler=profiler,
    callbacks=callbacks,
    min_epochs=epochs,
    max_epochs=epochs + 5,
    progress_bar_refresh_rate=20,
    weights_summary="top",
    benchmark=True,
    logger=tb_logger,
)

trainer.fit(model, train_loader, valid_loader)

final_checkpoint_file = os.path.join(exp_log_dir, "final_epoch.pth")
torch.save(model.state_dict(), final_checkpoint_file)

model.eval()
model = model.to(device)
df = validate(model, valid_loader, "valid", exp_log_dir)
df = validate(model, test_loader, "test", exp_log_dir)
