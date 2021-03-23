import os
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm.auto import tqdm

save_dir = "./results"


class TrackLosses(object):
    def __init__(self, name):
        self.reset()
        self.name = name

    def update(self, x):
        self.last_value = x
        self.values.append(x)
        self.average = mean(self.values)

    def reset(self):
        self.last_value = 0
        self.values = []
        self.average = 0

    def plot(self):
        plt.scatter(np.arange(0, len(self.values)), self.values, s=1, label=self.name)
        plt.title(label=self.name)
        plt.legend()


def show_images(
    images,
    save=False,
    fname=None,
    path=save_dir,
    nrow=4,
    title=None,
):
    images = images.detach().cpu()
    images = torchvision.utils.make_grid(images, nrow=nrow)
    show_image(images, save, fname, path, title)


def show_image(
    img,
    save=False,
    fname=None,
    path=save_dir,
    title=None,
):
    plt.imshow(img.permute(1, 2, 0), interpolation="bicubic")
    if title is not None:
        plt.title(title, loc="right")
    plt.xticks([])
    plt.yticks([])
    fig = plt.gcf()
    fig.set_size_inches(15, 9.5)
    if save:
        save_location = os.path.join(path, fname)
        fig.savefig(save_location, bbox_inches="tight")
    plt.show()


device = "cuda"


def evaluate(
    model,
    loader,
    num_images=4,
    save=False,
    fname=None,
    path=save_dir,
):
    with torch.no_grad():
        model.eval()
        x, y, _ = next(iter(loader))
        x = x.to(device)
        y = y.to(device)
        fake_y = model.generate(x)
        result = torch.cat((x[:num_images], fake_y[-1][:num_images], y[:num_images]))
        psnr, ssim_val = model.return_metrics(fake_y[-1], y)
        metric = "SSIM: %.4f PSNR: %.4f dB" % (ssim_val, psnr)
        show_images(result, save, fname, path, title=metric)
        print("SSIM: ", ssim_val)
        print("PSNR: ", psnr)


def plot_metrics(df):
    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(221)
    ax2 = f.add_subplot(222)
    ax.boxplot(df["PSNR"].values, vert=False)
    ax.set_title("PSNR")
    ax2.boxplot(df["SSIM"].values, vert=False)
    ax2.set_title("SSIM")

    ax3 = f.add_subplot(223)
    ax4 = f.add_subplot(224)
    ax3.hist(df["PSNR"].values, bins=100)
    ax3.set_title("")
    ax4.hist(df["SSIM"].values, bins=100)
    ax4.set_title("")
    plt.tight_layout()
    plt.show()


def compare_metrics(df):
    f = plt.figure(figsize=(10, 5))
    ax3 = f.add_subplot(121)
    ax4 = f.add_subplot(122)
    ax3.hist(df["Dataset PSNR"].values, bins=100, label="Dataset")
    ax3.hist(df["Model PSNR"].values, bins=100, label="Model", alpha=0.9)
    ax3.legend()
    ax4.hist(df["Dataset SSIM"].values, bins=100, label="Dataset")
    ax4.hist(df["Model SSIM"].values, bins=100, label="Model", alpha=0.9)
    ax4.legend()
    plt.tight_layout()
    plt.show()


def print_stats(name, model_loss, data_loss):
    print(name)
    print("Best- Model: %.4f Dataset: %.4f " % (max(model_loss.values), max(data_loss.values)))
    print("Average- Model: %.4f Dataset: %.4f " % (model_loss.average, data_loss.average))
    print("Worst- Model: %.4f Dataset: %.4f " % (min(model_loss.values), min(data_loss.values)))


def validate(model, loader):
    with torch.no_grad():

        model.eval()
        data_psnr_loss = TrackLosses("PSNR")
        data_ssim_loss = TrackLosses("SSIM")
        model_psnr_loss = TrackLosses("PSNR")
        model_ssim_loss = TrackLosses("SSIM")
        for x, y, _ in tqdm(loader, total=len(loader)):
            x = x.to(device)
            y = y.to(device)
            fake_y = model.generate(x)
            psnr, ssim_val = model.return_metrics(fake_y[-1], y)
            model_psnr_loss.update(psnr)
            model_ssim_loss.update(ssim_val)

            psnr, ssim_val = model.return_metrics(x, y)
            data_psnr_loss.update(psnr)
            data_ssim_loss.update(ssim_val)

        print_stats("PSNR", model_psnr_loss, data_psnr_loss)
        print_stats("SSIM", model_ssim_loss, data_ssim_loss)

        print("\n")
        zippedList = list(zip(model_psnr_loss.values, model_ssim_loss.values))
        model_df = pd.DataFrame(zippedList, columns=["PSNR", "SSIM"])

        zippedList = list(zip(data_psnr_loss.values, data_ssim_loss.values))
        data_df = pd.DataFrame(zippedList, columns=["PSNR", "SSIM"])

        df = pd.concat([model_df, data_df], axis=1)
        df.columns = ["Model PSNR", "Model SSIM", "Dataset PSNR", "Dataset SSIM"]
        compare_metrics(df)
        print(df.describe())
