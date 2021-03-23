import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import JRDR
from models import DerainCNN
from utils import (
    get_test_loader,
    get_train_valid_loader,
    train_transform,
    valid_transform,
    validate,
)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

dataset_dir = os.path.join("data", "JRDR")
test_dir = os.path.join("data", "JRDR")
train_data = JRDR(root=dataset_dir, transform=train_transform)
valid_data = JRDR(root=dataset_dir, transform=valid_transform)
test_data = JRDR(root=test_dir, split="test", transform=valid_transform)

train_loader, valid_loader = get_train_valid_loader(train_data, valid_data, show_sample=True)
test_loader = get_test_loader(test_data)

model = DerainCNN()

save_dir = "./checkpoints"
checkpoint_file = os.path.join(save_dir, "latest_epoch.pth")
best_model_dir = os.path.join(save_dir, "models")
best_model_file = "{Validation_PSNR}"
report_file = os.path.join(save_dir, "report.txt")


class SaveModelCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        torch.save(pl_module.state_dict(), checkpoint_file)
        validate(pl_module, valid_loader)


checkpoint_callback = ModelCheckpoint(
    period=1,
    dirpath=best_model_dir,
    filename=best_model_file,
    verbose=True,
    monitor="Validation_PSNR",
    mode="max",
)

early_stopping = pl.callbacks.EarlyStopping("Validation_PSNR", 0.001, 10, True, "max")
profiler = pl.profiler.AdvancedProfiler(report_file)

callbacks = [SaveModelCallback(), checkpoint_callback, early_stopping]

trainer = Trainer(
    gpus=1,
    profiler=profiler,
    callbacks=callbacks,
    min_epochs=300,
    progress_bar_refresh_rate=20,
    weights_summary="top",
    benchmark=True,
)

trainer.fit(model, train_loader, valid_loader)

final_checkpoint_file = os.path.join(save_dir, "final_epoch.pth")
torch.save(model.state_dict(), final_checkpoint_file)

model.eval()
df = validate(model, valid_loader)
df = validate(model, test_loader)
