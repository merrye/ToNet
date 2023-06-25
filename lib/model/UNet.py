import torch
from torch import nn
from torch.nn import functional as F

from .layers import Res1d

class DoubleConv(nn.Module):
    """U-net double convolution block: (CNN => ReLU => BN) * 2"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_batch_norm=False,
                 ):
        super(DoubleConv, self).__init__()
        block = []
        # block.append(nn.Conv2d(in_channels, out_channels,
        #                        kernel_size=3, stride=1, padding=1))
        block.append(nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        # block.append(nn.Conv2d(out_channels, out_channels,
        #                        kernel_size=3, stride=1, padding=1))
        block.append(nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class Encoder(nn.Module):
    """U-net encoder"""
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList(
            [DoubleConv(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class UpConv(nn.Module):
    """U-net Up-Conv layer. Can be real Up-Conv or bilinear up-sampling"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_mode='bilinear',
                 ):
        super(UpConv, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2, padding=0)
        elif up_mode == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(mode='linear', scale_factor=2,
                            align_corners=False),
                # nn.Conv2d(in_channels, out_channels, kernel_size=3,
                #           stride=1, padding=1))
                nn.Conv1d(in_channels, out_channels, kernel_size=3,
                          stride=1, padding=1))
        else:
            raise ValueError("No such up_mode")

    def forward(self, x):
        return self.up(x)

class Decoder(nn.Module):
    """U-net decoder, made of up-convolutions and CNN blocks.
    The cropping is necessary when 0-padding, due to the loss of
    border pixels in every convolution"""
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super(Decoder, self).__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [UpConv(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList(
            [DoubleConv(2*chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            try:
                enc_ftrs = encoder_features[i]
                x = torch.cat([x, enc_ftrs], dim=1)
            except RuntimeError:
                enc_ftrs = F.interpolate(encoder_features[i], size=(x.size(-1)), mode="linear", align_corners=False)
                x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self,
                enc_chs=(3, 16, 32, 64, 128, 256),
                dec_chs=(256, 128, 64, 32, 16),
                out_chs=2,
                retain_dim=True):
        super(UNet, self).__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.output = Res1d(dec_chs[-1], out_chs, norm="GN", ng=1)
        self.retain_dim = retain_dim
    
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.output(out)

        return out[:, :, -1]

class UNetRes(nn.Module):
    def __init__(self,
                enc_chs=(3, 16, 32, 64, 128, 256),
                dec_chs=(256, 128, 64, 32, 16),
                out_chs=2):
        super(UNetRes, self).__init__()
        self.unet1 = UNet(enc_chs=enc_chs, dec_chs=dec_chs, out_chs=out_chs)
        self.unet2 = UNet(enc_chs=enc_chs, dec_chs=dec_chs, out_chs=out_chs)
    
    def forward(self, x):
        out = torch.cat([self.unet1(x), self.unet2(x)], dim=-1)
        return out