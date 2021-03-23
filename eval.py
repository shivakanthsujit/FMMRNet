import glob
import os

import albumentations as A
import numpy as np
import PIL
import torch
from albumentations.pytorch import ToTensorV2

from models import DerainCNN
from utils import show_image, show_images

device = "cuda" if torch.cuda.is_available() else "cpu"
image_filepath = os.path.join("images", "Original.png")
results_dir = "./results"
image_results_dir = os.path.join(results_dir, "multiresolution")


def get_valid_transforms():
    return A.Compose(
        [A.Resize(256, 256), A.ToFloat(255), ToTensorV2()],
        additional_targets={"image1": "image"},
    )


def load_transform_image(path=image_filepath, device="cuda"):
    rain_img = PIL.Image.open(path)
    rain_img = np.array(rain_img)
    aug = valid_transform(image=rain_img, image1=rain_img)
    rain_img = aug["image"]
    rain_img = rain_img.to(device).unsqueeze(0)
    return rain_img


model_dir = os.path.join("checkpoints", "final_epoch.pth")
model = DerainCNN()
model.load_state_dict(torch.load(model_dir))
model = model.to(device)
model.eval()


with torch.no_grad():
    x = load_transform_image()
    d_y, d_z = model.generator(x)
    dy = [torch.clamp(imgy, 0.0, 1.0) for imgy in d_y]
    for i in range(len(dy)):
        img = dy[i]
        fname = f"Derained {i}.png"
        show_image(img[0].cpu(), save=True, fname=fname, path=image_results_dir)

        img = d_z[i]
        fname = f"Residual {i}.png"
        show_image(img[0].cpu(), save=True, fname=fname, path=image_results_dir)

valid_transform = get_valid_transforms()
rain_img_paths = glob.glob("./images/syn/*_in.png")
clear_image_paths = glob.glob("./images/syn/*_GT.png")

for i in range(len(rain_img_paths)):
    rain_img_path = rain_img_paths[i]
    clear_image_path = clear_image_paths[i]
    rain_img = load_transform_image(rain_img_path)
    norain_img = load_transform_image(clear_image_path)

    fake_y = model.generate(rain_img.unsqueeze(0))
    result = torch.cat((rain_img.unsqueeze(0), fake_y[-1], norain_img.unsqueeze(0)))
    psnr, ssim_val = model.return_metrics(fake_y[-1], norain_img.unsqueeze(0))

    output_file_dir = os.path.join(results_dir, "outputs")
    metric = f"SSIM: {ssim_val:.4f} PSNR: {psnr:.4f} dB"
    fname = os.path.split(rain_img_path)[1].split("_")[0] + ".png"
    show_images(result, save=True, fname=fname, path=output_file_dir, title=metric)

imgs = [1, 2, 3]
if imgs is not None:
    img_files = [os.path.join("images", f"{i}.png") for i in imgs]
else:
    img_files = glob.glob(os.path.join("images", "*.png"))

for path in img_files:
    rain_img = load_transform_image(path)
    fake_y = model.generate(rain_img.unsqueeze(0))
    result = torch.cat((rain_img.unsqueeze(0), fake_y[-1]))
    fname = f"Derained {os.path.split(path)[1]}"
    show_images(result, save=True, fname=fname, path=output_file_dir)
