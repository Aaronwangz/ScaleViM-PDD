from vm0 import  UpConv,PDU,CALayer,ChannelPixelAttention,CP_Attention_block
import torch
from vm0 import AttentionModule
import torch.nn as nn
from functools import partial
import math
import torch.nn.functional as F


class FM(nn.Module):
    def __init__(self, nc):
        """
        Fourier Processing Module

        """
        super(FM, self).__init__()

        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.cat = nn.Conv2d(nc, nc, 1, 1, 0)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out = self.cat(x_out)
        return x_out + x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Enhanced_EAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=16):
        super().__init__()
        valid_groups = min(num_groups, F_int // 4)

        self.g_conv = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, groups=valid_groups),
            nn.BatchNorm2d(F_int),
            nn.GELU()
        )

        self.x_conv = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, groups=valid_groups),
            nn.BatchNorm2d(F_int),
            nn.GELU()
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_l, 1),  # 多通道注意力
            nn.BatchNorm2d(F_l),
            nn.Sigmoid(),
            FM(F_l)
        )

    def forward(self, g, x):
        g_proj = self.g_conv(g)
        x_proj = self.x_conv(x)
        attn = self.psi(g_proj + x_proj)
        return x * attn + x


class SPI_Attention(nn.Module):
    def __init__(self, in_dim, kernel_size=3, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.eag = Enhanced_EAG(in_dim, in_dim, in_dim // 2)
            self.fusion = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                CP_Attention_block(default_conv, in_dim, kernel_size),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                CP_Attention_block(default_conv, in_dim, kernel_size),
            )

    def forward(self, x, skip=None):
        if not self.is_bottom:
            if skip.size()[-2:] != x.size()[-2:]:
                skip = F.interpolate(skip, x.shape[2:], mode='bilinear', align_corners=True)
            x = self.eag(x, skip)
        return self.fusion(x)


class NNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(NNBlock, self).__init__()


        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.pud = PDU(ch_out)
        self.attention_module = AttentionModule(in_channels=ch_out, filters=ch_out // 4, reduction_ratio=4,
                                                dim=dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand,
                                                state_dim=state_dim)


    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pud(out)
        out = self.attention_module(out)

        return out + residual


class U_Net(nn.Module):
    def __init__(self, dim=32):
        super(U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = NNBlock(3, dim, dim=dim)
        self.Conv2 = NNBlock(dim, dim*2, dim=dim*2)
        self.Conv3 = NNBlock(dim*2, dim*4, dim=dim*4)
        self.Conv4 = NNBlock(dim*4, dim*8, dim=dim*8)
        self.Conv5 = NNBlock(dim*8, dim*16, dim=dim*16)

        # Decoding path
        self.Up5 = UpConv(dim*16, dim*8)
        self.Up_conv5 = NNBlock(dim*8, dim*8, dim=dim*8)

        self.Up4 = UpConv(dim*8, dim*4)
        self.Up_conv4 = NNBlock(dim*4, dim*4, dim=dim*4)

        self.Up3 = UpConv(dim*4, dim*2)
        self.Up_conv3 = NNBlock(dim*2, dim*2, dim=dim*2)

        self.Up2 = UpConv(dim*2, dim)
        self.Up_conv2 = NNBlock(dim, dim, dim=dim)

        # Final 1x1 Convolution to get the desired number of classes
        self.Final_conv = nn.Conv2d(dim, 3, kernel_size=1, stride=1, padding=0)


        self.eff1 = SPI_Attention(in_dim=dim * 8)
        self.eff2 = SPI_Attention(in_dim=dim * 4)
        self.eff3 = SPI_Attention(in_dim=dim * 2)
        self.eff4 = SPI_Attention(in_dim=dim)




        self.adjust_channel4 = nn.Conv2d(dim*8, dim*8, kernel_size=1, stride=1, padding=0, bias=True)
        self.adjust_channel3 = nn.Conv2d(dim*4, dim*4, kernel_size=1, stride=1, padding=0, bias=True)
        self.adjust_channel2 = nn.Conv2d(dim*2, dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        self.adjust_channel1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Encoding Path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)


        # Decoding Path
        d5 = self.Up5(x5)
        d5 = self.eff1(x4, d5) + self.adjust_channel4(x4)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.eff2(x3, d4) + self.adjust_channel3(x3)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.eff3(x2, d3) + self.adjust_channel2(x2)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.eff4(x1, d2) + self.adjust_channel1(x1)
        d2 = self.Up_conv2(d2)

        # Final Output
        output = self.Final_conv(d2)

        output = output + x
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = U_Net(dim=16).to(device)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print("Output size:", output.size())

