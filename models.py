from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from pytorch_msssim import ssim as pyssim
from torch import nn, optim
from torch.nn.utils import spectral_norm

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
vgg19 = nn.Sequential()
vgg19.add_module("normalise", normalization)
i = 0
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = "conv_{}".format(i)
    elif isinstance(layer, nn.ReLU):
        name = "relu_{}".format(i)
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = "pool_{}".format(i)
    elif isinstance(layer, nn.BatchNorm2d):
        name = "bn_{}".format(i)
    else:
        raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

    if name != "pool_4":
        vgg19.add_module(name, layer)
    else:
        break


def cal_perpetual_loss(vae_image, ori_image):
    x = vgg19.normalise(vae_image)
    y = vgg19.normalise(ori_image)
    x = vgg19.relu_1(vgg19.conv_1(x))
    y = vgg19.relu_1(vgg19.conv_1(y))
    l1 = F.mse_loss(x, y)
    # l1 = F.l1_loss(x, y)
    x = vgg19.relu_2(vgg19.conv_2(x))
    y = vgg19.relu_2(vgg19.conv_2(y))
    l2 = F.mse_loss(x, y)
    # l2 = F.l1_loss(x, y)
    x = vgg19.relu_3(vgg19.conv_3(vgg19.pool_2(x)))
    y = vgg19.relu_3(vgg19.conv_3(vgg19.pool_2(y)))
    l3 = F.mse_loss(x, y)
    # l3 = F.l1_loss(x, y)

    return l1 + l2 + l3


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=(0, 0),
        dilation=1,
        groups=1,
        relu=True,
        norm=False,
        bias=False,
        spec_norm=True,
    ):
        super(Conv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if spec_norm:
            self.conv = spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
            )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.relu:
            x = self.relu(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm=False,
        spec_norm=True,
    ):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        if spec_norm:
            self.conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation))
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.norm:
            out = self.norm(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduce=4):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels // reduce, kernel_size=1, stride=1)
        self.conv2 = ConvLayer(out_channels // reduce, out_channels // reduce, kernel_size=3, stride=1)
        self.conv3 = ConvLayer(out_channels // reduce, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.add(out, x)

        return out


# class FMGCBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):

#         super(FMGCBlock, self).__init__()

#         self.conv_init = nn.Sequential(
#             ConvLayer(in_channels, out_channels, kernel_size=1, stride=1),
#             ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
#         )

#         self.branch0 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
#         self.branch1 = nn.Sequential(
#             Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
#             ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
#         )
#         self.branch2 = nn.Sequential(
#             Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
#             Conv(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
#             Conv(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
#             ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
#         )
#         self.branch3 = nn.Sequential(
#             Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
#             ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
#             ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
#         )

#         self.branch4 = nn.Sequential(
#             Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
#             Conv(out_channels, out_channels, kernel_size=(7, 1), stride=1, padding=(3, 0)),
#             Conv(out_channels, out_channels, kernel_size=(1, 7), stride=1, padding=(0, 3)),
#         )

#         self.conv = Bottleneck(out_channels, out_channels)


#     def forward(self, x):

#         x0 = self.branch0(x)
#         x = self.conv_init(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x4 = self.branch4(x)

#         x = torch.add(x0, x1)
#         x = torch.add(x2, x)
#         x = torch.add(x3, x)
#         x = torch.add(x4, x)
#         x = self.conv(x)

#         return x

def get_mutliscale_branch(in_channels, out_channels, kernel):
    assert kernel in [1, 3, 5, 7], f"Invalid kernel size {kernel}"
    if kernel == 1:
        return nn.Sequential(
            Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
        )
    elif kernel == 3:
        return nn.Sequential(
            Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            Conv(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            Conv(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
        )
    elif kernel == 5:
        return nn.Sequential(
            Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
        )
    elif kernel == 7:
        return nn.Sequential(
            Conv(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            Conv(out_channels, out_channels, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            Conv(out_channels, out_channels, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )


class FMGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, multiscale_kernels=[1, 3, 5, 7]):

        super(FMGCBlock, self).__init__()

        self.skip = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv_init = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size=1, stride=1),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
        )
        self.branches = nn.ModuleList([get_mutliscale_branch(in_channels, out_channels, kernel) for kernel in multiscale_kernels])
        self.bottleneck = Bottleneck(out_channels, out_channels)


    def forward(self, x):
        x0 = self.skip(x)
        x = self.conv_init(x)
        for branch in self.branches:
            x0 += branch(x)
        x = self.bottleneck(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):

        super(ResidualBlock, self).__init__()
        self.pading = nn.ReflectionPad2d(1)
        self.conv_init = ConvLayer(in_channels, in_channels, kernel_size=1, stride=stride)
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=stride)

    def forward(self, x):

        residul = self.conv_init(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.add(out, residul)

        return out


class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        y = x1 + x2
        y = self.conv_du(y)
        return x * y


class PixelAttention(nn.Module):
    def __init__(self, nf):

        super(PixelAttention, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, attention_type="channel", reduction=16):
        super(ConvTranspose2d, self).__init__()

        layers = [ResidualBlock(out_channels, out_channels)]
        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*layers)
        self.up = Upsample(in_channels, out_channels)
        if attention_type == "channel":
            self.attn = CALayer(out_channels, reduction=reduction)
        elif attention_type == "pixel":
            self.attn = PixelAttention(out_channels)
        else:
            raise NotImplementedError(attention_type)

    def forward(self, x, skip_x):
        x = self.up(x)
        skip_x = self.attn(skip_x)
        x = torch.add(x, skip_x)
        x = self.model(x)
        return x


class RGBConv(nn.Module):
    def __init__(self, in_channels):
        super(RGBConv, self).__init__()
        self.clear = nn.Sequential(Conv(in_channels, 3, 1, relu=False, norm=False))

    def forward(self, x):
        clear = self.clear(x)
        return clear


class FMMRNet(nn.Module):
    def __init__(self, depth=5, m=1):
        super(FMMRNet, self).__init__()

        self.init = nn.Sequential(
            Conv(3, 8, 1),
            ConvLayer(8, 16, kernel_size=3, stride=1),
            ConvLayer(16, 16, kernel_size=3, stride=1),
        )
        self.encode1 = nn.Sequential(FMGCBlock(16, 16), ConvLayer(16, 32, kernel_size=3, stride=2))
        self.encode2 = nn.Sequential(FMGCBlock(32, 32), ConvLayer(32, 64, kernel_size=3, stride=2))
        self.encode3 = nn.Sequential(FMGCBlock(64, 64), ConvLayer(64, 128, kernel_size=3, stride=2))
        self.encode4 = nn.Sequential(FMGCBlock(128, 128), ConvLayer(128, 256, kernel_size=3, stride=2))
        self.encode5 = nn.Sequential(FMGCBlock(256, 256), ConvLayer(256, 256, kernel_size=3, stride=2))

        self.center = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )

        self.decode1 = ConvTranspose2d(256, 256)
        self.decode2 = ConvTranspose2d(256, 128)
        self.decode3 = ConvTranspose2d(128, 64)
        self.decode4 = ConvTranspose2d(64, 32)
        self.decode5 = ConvTranspose2d(32, 16)

        self.rgb_conv1 = RGBConv(256)
        self.rgb_conv2 = RGBConv(128)
        self.rgb_conv3 = RGBConv(64)
        self.rgb_conv4 = RGBConv(32)
        self.rgb_conv5 = RGBConv(16)

    def downsample(self, input):
        input = [input] + [F.avg_pool2d(input, int(np.power(2, i))) for i in range(1, 5)]
        return list(reversed(input))

    def forward(self, x):
        x1 = self.init(x)
        d1 = self.encode1(x1)
        d2 = self.encode2(d1)
        d3 = self.encode3(d2)
        d4 = self.encode4(d3)
        d5 = self.encode5(d4)

        out = self.center(d5)

        u1 = self.decode1(out, d4)
        u2 = self.decode2(u1, d3)
        u3 = self.decode3(u2, d2)
        u4 = self.decode4(u3, d1)
        u5 = self.decode5(u4, x1)

        u1_clear = self.rgb_conv1(u1)
        u2_clear = self.rgb_conv2(u2)
        u3_clear = self.rgb_conv3(u3)
        u4_clear = self.rgb_conv4(u4)
        u5_clear = self.rgb_conv5(u5)

        down = self.downsample(x)
        res = [u1_clear, u2_clear, u3_clear, u4_clear, u5_clear]
        clear = [down[i] - res[i] for i in range(len(down))]
        return clear, res


class FMMRNetModular(nn.Module):
    def __init__(self, input_size=256, channel_mul=16, depth=5, center_depth=4, attention_type="channel", reduction=16, multiscale_kernels=[1, 3, 5, 7]):
        super(FMMRNetModular, self).__init__()

        assert input_size >= np.power(2, depth), "Latent size will diminish to zero."
        assert channel_mul >= reduction, "For channel attention, the reduction <= channel_mul"

        self.channel_mul = channel_mul
        self.depth = depth
        self.init = nn.Sequential(
            Conv(3, self.channel_mul // 2, 1),
            ConvLayer(self.channel_mul // 2, self.channel_mul, kernel_size=3, stride=1),
            ConvLayer(self.channel_mul, self.channel_mul, kernel_size=3, stride=1),
        )

        encoder_blocks = []
        decoder_blocks = []

        for i in range(self.depth):
            # Make encoder block
            dim1 = self.channel_mul * np.power(2, i)
            dim2 = self.channel_mul * np.power(2, i + 1) if i != self.depth - 1 else self.channel_mul * np.power(2, i)
            encoder_blocks.append(nn.Sequential(FMGCBlock(dim1, dim1, multiscale_kernels), ConvLayer(dim1, dim2, kernel_size=3, stride=2)))

            # Make corresponding decoder block
            dim3 = self.channel_mul * np.power(2, self.depth - i)
            dim4 = dim3 // 2
            dim3 = self.channel_mul * np.power(2, self.depth - i - 1) if i == 0 else dim3
            decoder_blocks.append(ConvTranspose2d(dim3, dim4, attention_type=attention_type, reduction=reduction))

        self.encoder = nn.ModuleList(encoder_blocks)
        self.center_block = nn.Sequential(
            *[
                ResidualBlock(self.channel_mul * np.power(2, self.depth - 1), self.channel_mul * np.power(2, self.depth - 1))
                for _ in range(center_depth)
            ]
        )
        self.decoder = nn.ModuleList(decoder_blocks)
        self.rgb_convs = nn.ModuleList(
            [RGBConv(self.channel_mul * np.power(2, self.depth - i - 1)) for i in range(self.depth)]
        )

    def downsample(self, input):
        input = [input] + [F.avg_pool2d(input, int(np.power(2, i))) for i in range(1, self.depth)]
        return list(reversed(input))

    def forward(self, x):
        x1 = self.init(x)

        encoder_outputs = [x1]
        for i in range(self.depth):
            encoder_out = self.encoder[i](encoder_outputs[-1])
            encoder_outputs.append(encoder_out)

        out = self.center_block(encoder_outputs[-1])

        decoder_outputs = [out]
        residual_outputs = []
        for i in range(self.depth):
            decoder_output = self.decoder[i](decoder_outputs[i], encoder_outputs[-2 - i])
            rgb_output = self.rgb_convs[i](decoder_output)
            decoder_outputs.append(decoder_output)
            residual_outputs.append(rgb_output)

        downsampled_inputs = self.downsample(x)
        clear_outputs = [downsampled_inputs[i] - residual_outputs[i] for i in range(self.depth)]
        return clear_outputs, residual_outputs


class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()

    def downsample(self, input):
        input = [input] + [F.avg_pool2d(input, int(np.power(2, i))) for i in range(1, 5)]
        return list(reversed(input))

    def get_psnr(self, fake_y, y):
        temp = F.mse_loss(fake_y, y)
        psnr = -10 * torch.log10(temp)
        return psnr

    def get_ssim(self, fake_y, y):
        ssim = pyssim(y, fake_y, data_range=1.0)
        return ssim

    def return_metrics(self, input, target):
        psnr = self.get_psnr(input, target)
        ssim_val = self.get_ssim(input, target)
        return psnr, ssim_val

    @abstractmethod
    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass


class DerainCNN(CNN):
    def __init__(self, lr=1e-4, beta1=0.5, beta2=0.999, gamma=0.8):
        super(DerainCNN, self).__init__()

        self.model = FMMRNet()
        self.model.apply(weight_init)

        self.img_size = 256
        self.imgs = []

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma

    def pixel_loss(self, fake_y, y):
        return F.mse_loss(fake_y, y)

    def multiscale_loss(self, fake_y, y):
        loss_pixel = 0
        for i in range(len(fake_y)):
            loss_pixel += (i + 1) * (self.pixel_loss(fake_y[i], y[i]))
        return loss_pixel

    def configure_optimizers(self):
        optimizer_g = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return optimizer_g

    def forward(self, x):
        d_y, _ = self.model(x)
        return d_y, d_y

    def loss_function(self, fake_y, y):
        loss_pixel = self.multiscale_loss(fake_y, y)
        loss_per = cal_perpetual_loss(fake_y[-1], y[-1])
        loss_gen = loss_pixel + self.gamma * loss_per
        return loss_gen, loss_pixel, loss_per

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = self.downsample(y)
        fake_y, _ = self(x)

        # if batch_idx % 50 == 0:
        #     result = torch.cat((x[:1], fake_y[-1][:1], y[-1][:1]))
        #     grid = torchvision.utils.make_grid(result)
        #     self.logger.experiment.add_image("train_images", grid)

        loss_gen, loss_pixel, loss_per = self.loss_function(fake_y, y)
        psnr, loss_ssim = self.return_metrics(fake_y[-1], y[-1])

        self.log("train/Total_Loss", loss_gen)
        self.log("train/Pixel_Loss", loss_pixel)
        self.log("train/PSNR_Loss", psnr, prog_bar=True)
        self.log("train/Perceptual_Loss", loss_per)
        self.log("train/SSIM_Loss", loss_ssim)
        return loss_gen

    def generate(self, x):
        d_y, _ = self.model(x)
        dy = [torch.clamp(imgy, 0.0, 1.0) for imgy in d_y]
        return dy

    def validation_step(self, batch, batch_id):
        x, y, _ = batch
        fake_y = self.generate(x)
        psnr, ssim = self.return_metrics(fake_y[-1], y)
        result = torch.cat((x[:1], fake_y[-1][:1], y[:1]))
        grid = torchvision.utils.make_grid(result)
        self.imgs = grid
        return {"val_acc": psnr, "PSNR": psnr, "SSIM": ssim}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([x["val_acc"] for x in outputs]).mean()
        avg_psnr = torch.FloatTensor([x["PSNR"] for x in outputs]).mean()
        avg_ssim = torch.FloatTensor([x["SSIM"] for x in outputs]).mean()

        self.log("valid/PSNR", avg_psnr)
        self.log("valid/SSIM", avg_ssim)
        self.logger.experiment.add_image("valid_images", self.imgs)
        return avg_loss


class DerainCNNModular(CNN):
    def __init__(
        self,
        model,
        input_size=256,
        gamma=0.8,
        optimizer="Adam",
        lr=1e-4,
        beta1=0.5,
        beta2=0.999,
    ):
        super().__init__()

        self.model = model
        self.model.apply(weight_init)

        self.img_size = input_size
        self.imgs = []
        self.example_input_array = torch.rand(1, 3, self.img_size, self.img_size)

        self.gamma = gamma

        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def pixel_loss(self, fake_y, y):
        return F.mse_loss(fake_y, y)

    def multiscale_loss(self, fake_y, y):
        loss_pixel = 0
        for i in range(len(fake_y)):
            loss_pixel += (i + 1) * (self.pixel_loss(fake_y[i], y[i]))
        return loss_pixel

    def configure_optimizers(self):
        optimizer_g = self.optimizer_class(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return optimizer_g

    def forward(self, x):
        return self.model(x)

    def loss_function(self, fake_y, y):
        loss_pixel = self.multiscale_loss(fake_y, y)
        loss_per = cal_perpetual_loss(fake_y[-1], y[-1])
        loss_gen = loss_pixel + self.gamma * loss_per
        return loss_gen, loss_pixel, loss_per

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = self.downsample(y)
        fake_y, _ = self(x)

        if batch_idx % 50 == 0:
            result = torch.cat((x[:1], fake_y[-1][:1], y[-1][:1]))
            grid = torchvision.utils.make_grid(result)
            self.logger.experiment.add_image("train_images", grid)

        loss_gen, loss_pixel, loss_per = self.loss_function(fake_y, y)
        psnr, ssim = self.return_metrics(fake_y[-1], y[-1])

        self.log("train/loss/total", loss_gen, on_step=False, on_epoch=True)
        self.log("train/loss/pixel", loss_pixel, on_step=False, on_epoch=True)
        self.log("train/loss/perceptual", loss_per, on_step=False, on_epoch=True)
        self.log("train/PSNR", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/SSIM", ssim, on_step=False, on_epoch=True)
        return loss_gen

    def generate(self, x):
        d_y, _ = self(x)
        dy = [torch.clamp(imgy, 0.0, 1.0) for imgy in d_y]
        return dy

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        fake_y = self.generate(x)
        psnr, ssim = self.return_metrics(fake_y[-1], y)
        result = torch.cat((x[:1], fake_y[-1][:1], y[:1]))
        grid = torchvision.utils.make_grid(result)
        self.imgs = grid
        self.log("valid/PSNR", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid/SSIM", ssim, on_step=False, on_epoch=True, prog_bar=False)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_image("valid_images", self.imgs)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        fake_y = self.generate(x)
        psnr, ssim = self.return_metrics(fake_y[-1], y)
        result = torch.cat((x[:1], fake_y[-1][:1], y[:1]))
        grid = torchvision.utils.make_grid(result)
        self.test_imgs = grid
        self.log("test/PSNR", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/SSIM", ssim, on_step=False, on_epoch=True, prog_bar=False)

    def test_epoch_end(self, outputs):
        self.logger.experiment.add_image("Test_Set_Images", self.imgs)
