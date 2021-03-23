import glob
import os

import albumentations as A
import numpy as np
import PIL
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataloader import DataLoader

from utils import show_images


def get_train_transforms():
    return A.Compose(
        [
            A.RandomCrop(256, 256),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.9,
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.9),
                ],
                p=0.9,
            ),
            A.ToFloat(255),
            ToTensorV2(),
        ],
        additional_targets={"image1": "image"},
    )


def get_valid_transforms():
    return A.Compose(
        [A.CenterCrop(256, 256), A.ToFloat(255), ToTensorV2()],
        additional_targets={"image1": "image"},
    )


train_transform = get_train_transforms()
valid_transform = get_valid_transforms()


class JRDR(torch.utils.data.Dataset):
    def __init__(self, root, type="Light", split="train", transform=train_transform):
        self.root = root
        self.data_dir = os.path.join(self.root, "rain_data_" + split + "_" + type)

        if type == "Heavy" or split == "test":
            self.rain_dir = os.path.join(self.data_dir, "rain/X2")
        else:
            self.rain_dir = os.path.join(self.data_dir, "rain")

        self.norain_dir = os.path.join(self.data_dir, "norain")
        self.files = glob.glob(self.rain_dir + "/*.*")
        if len(self.files) == 0:
            raise RuntimeError("Dataset not found.")

        self.transform = transform

    def get_file_name(self, idx):
        img1 = self.files[idx]
        _, img2 = os.path.split(img1)
        img2 = img2.split("x2")[0] + ".png"
        img2 = os.path.join(self.norain_dir, img2)
        return img1, img2

    def __getitem__(self, idx):
        img1, img2 = self.get_file_name(idx)
        rain_img = PIL.Image.open(img1)
        norain_img = PIL.Image.open(img2)
        if self.transform is not None:
            rain_img, norain_img = np.array(rain_img), np.array(norain_img)
            aug = self.transform(image=rain_img, image1=norain_img)
            rain_img, norain_img = aug["image"], aug["image1"]
        return rain_img, norain_img, rain_img - norain_img

    def __len__(self):
        return len(glob.glob(self.norain_dir + "/*.*"))


class li_cvpr(torch.utils.data.Dataset):
    def __init__(self, root, transform=valid_transform):
        self.root = root
        self.rain_files = sorted(glob.glob(self.root + "/*in.png"))
        self.norain_files = sorted(glob.glob(self.root + "/*GT.png"))
        if len(self.rain_files) == 0 or len(self.norain_files) == 0:
            raise RuntimeError("Dataset not found.")
        self.transform = transform

    def get_file_name(self, idx):
        img1 = self.rain_files[idx]
        img2 = self.norain_files[idx]
        return img1, img2

    def __getitem__(self, idx):
        img1, img2 = self.get_file_name(idx)
        rain_img = PIL.Image.open(img1)
        norain_img = PIL.Image.open(img2)
        if self.transform is not None:
            rain_img, norain_img = np.array(rain_img), np.array(norain_img)
            aug = self.transform(image=rain_img, image1=norain_img)
            rain_img, norain_img = aug["image"], aug["image1"]
        return rain_img, norain_img, rain_img - norain_img

    def __len__(self):
        return len(self.rain_files)


class IDGAN(torch.utils.data.Dataset):
    def __init__(self, root, split="train", syn=True, transform=train_transform):
        self.root = root
        self.data_dir = os.path.join(self.root, "rain")

        if split == "test":
            self.rain_dir = os.path.join(self.data_dir, "test_syn")
        else:
            self.rain_dir = os.path.join(self.data_dir, "training")

        self.norain_dir = self.rain_dir
        self.files = glob.glob(self.rain_dir + "/*.*")
        if len(self.files) == 0:
            raise RuntimeError("Dataset not found.")

        self.transform = transform

    def get_file_name(self, idx):
        img1 = self.files[idx]
        _, img2 = os.path.split(img1)
        img2 = img2.split("x2")[0] + ".png"
        img2 = os.path.join(self.norain_dir, img2)
        return img1, img2

    def __getitem__(self, idx):
        img1 = self.files[idx]
        im = PIL.Image.open(img1)
        w, h = im.size
        norain_img = im.crop((0, 0, w // 2, h))
        norain_img = np.array(norain_img)
        rain_img = im.crop((w // 2, 0, w, h))
        rain_img = np.array(rain_img)
        if self.transform is not None:
            rain_img, norain_img = np.array(rain_img), np.array(norain_img)
            aug = self.transform(image=rain_img, image1=norain_img)
            rain_img, norain_img = aug["image"], aug["image1"]
        return rain_img, norain_img, rain_img - norain_img

    def __len__(self):
        return len(glob.glob(self.norain_dir + "/*.*"))


def get_train_valid_loader(
    train_data,
    valid_data,
    batch_size=4,
    valid_size=0.1,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
    shuffle=True,
    seed=42,
):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_dataset = torch.utils.data.Subset(train_data, train_idx)
    valid_dataset = torch.utils.data.Subset(valid_data, valid_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("Training Batches: ", len(train_loader))
    print("Validation Batches: ", len(valid_loader))

    # visualize some images
    if show_sample:
        x, y, z = next(iter(train_loader))
        show_images(torch.cat((x, y, z)))
        x, y, z = next(iter(valid_loader))
        show_images(torch.cat((x, y, z)))

    return train_loader, valid_loader


def get_test_loader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=False):

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    print("Testing Batches: ", len(test_loader))

    return test_loader
