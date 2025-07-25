import math
import torch
import torch.nn as nn
from timm.layers import SqueezeExcite
import torch.nn.functional as F


class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


class Stem(nn.Module):
    def __init__(self, in_dim=3, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, dim // 8, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None))

    def forward(self, x):
        x = self.conv(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, hidden_dim, kernel_size=1),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim),
            SqueezeExcite(hidden_dim, .25),
            ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None)
        )

        self.dwconv1 = ConvLayer2D(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L = x.shape
        H = int(math.sqrt(L))
        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, H)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)

        AB = (A * B)
        h = x @ AB.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C  # B C N, B C L -> B C L

        y = y.view(batch, -1, H, H).contiguous()  # + x * self.D  # B C H W
        return y, h


class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)

        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # DWconv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # HSM-SSD
        x_prev = x
        x, h = self.mixer(self.norm(x.flatten(2)))
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        return x


class MSM(nn.Module):


    def __init__(self, in_channels, filters):
        super(MSM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        return torch.cat([o1, o2, o3, o4], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class CAM(nn.Module):


    def __init__(self, in_channels, reduction_ratio=4):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Sigmoid(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.module(x)
        attention_weights = attention_weights.unsqueeze(2).unsqueeze(3)
        return x * attention_weights.expand_as(x)


class AttentionModule(nn.Module):


    def __init__(self, in_channels, filters, reduction_ratio=4, dim=64, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(AttentionModule, self).__init__()
        self.inc = MSM(in_channels=in_channels, filters=filters)
        self.adjust_channel = nn.Conv2d(filters * 4, in_channels, kernel_size=1, stride=1, padding=0, bias=True)


        self.eff_vim_block = EfficientViMBlock(dim=dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim)

    def forward(self, x):
        inc_out = self.inc(x)
        adjusted_out = self.adjust_channel(inc_out)


        out = self.eff_vim_block(adjusted_out)

        return out


class NNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(NNBlock, self).__init__()


        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)


        self.attention_module = AttentionModule(in_channels=ch_out, filters=ch_out // 4, reduction_ratio=4,
                                                dim=dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand,
                                                state_dim=state_dim)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)


        out = self.attention_module(out)


        return out + residual


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


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
            nn.Conv2d(F_int, F_l, 1),
            nn.BatchNorm2d(F_l),
            nn.Sigmoid()
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
                CP_Attention_block(default_conv, in_dim, kernel_size)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                CP_Attention_block(default_conv, in_dim, kernel_size)
            )

    def forward(self, x, skip=None):
        if not self.is_bottom:
            if skip.size()[-2:] != x.size()[-2:]:
                skip = F.interpolate(skip, x.shape[2:], mode='bilinear', align_corners=True)
            x = self.eag(x, skip)
        return self.fusion(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_value = self.avg_pool(x).view(x.size(0), -1)
        max_value = self.max_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_value + max_value).view(x.size(0), x.size(1), 1, 1)
        return channel_attention


class PixelAttention(nn.Module):
    def __init__(self):
        super(PixelAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)

        pixel_attention = self.conv1(combined)
        pixel_attention = self.sigmoid(pixel_attention)

        return pixel_attention


class ChannelPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelPixelAttention, self).__init__()

        self.channel_attention = ChannelAttention(in_channels)
        self.pixel_attention = PixelAttention()

    def forward(self, x):
        channel_att = self.channel_attention(x)
        pixel_att = self.pixel_attention(x)
        return channel_att * pixel_att * x


class PDU(nn.Module):
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid())
        self.td = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.ka(a)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j

class PBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(PBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.pdu = PDU(ch_out)
        self.CPA = ChannelPixelAttention(ch_out)
        self.residual_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.CPA(out)
        out = self.pdu(out)

        return out + residual

if __name__ == '__main__':
    block = EfficientViMBlock(dim=96).to('cuda')

    input_tensor = torch.rand(4, 96, 32, 32).to('cuda')

    output = block(input_tensor)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())
