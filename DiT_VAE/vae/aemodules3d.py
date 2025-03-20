# TATS
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_vae import MultiHeadAttention


def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group',
                 padding_type='replicate'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())

        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)
        self.weight = self.conv.weight

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x
class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            Normalize(n_hiddens),
            SiLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            Normalize(n_hiddens // 2),
            SiLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            Normalize(n_hiddens),
            SiLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, z_channels, double_z, image_channel=3, norm_type='group',
                 padding_type='replicate', res_num=1):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()
        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2 ** i
            out_channels = n_hiddens * 2 ** (i + 1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            stride = list(stride)
            stride[0] = 1
            stride = tuple(stride)
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)

            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU(),
            SamePadConv3d(out_channels, 2 * z_channels if double_z else z_channels,
                          kernel_size=3,
                          stride=1,
                          padding_type=padding_type)
        )
        self.out_channels = out_channels


    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, z_channels, image_channel, norm_type='group'):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        in_channels = z_channels
        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens * 2 ** (max_us - i + 1)
            out_channels = n_hiddens * 2 ** (max_us - i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            us = list(us)
            us[0] = 1
            us = tuple(us)
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_out = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = x
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_out(h)
        return h


class EncoderRe(nn.Module):
    def __init__(self, n_hiddens, downsample, z_channels, double_z, image_channel=3, norm_type='group',
                 padding_type='replicate', n_res_layers=2):
        super().__init__()
        # n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        # max_ds = n_times_downsample.max()
        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i, step in enumerate(downsample):
            block = nn.Module()
            in_channels = n_hiddens
            out_channels = n_hiddens
            stride = [1, downsample[i], downsample[i]]
            stride = tuple(stride)
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)


        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(out_channels)
              for _ in range(n_res_layers)]
        )
        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU(),
            SamePadConv3d(out_channels, 2 * z_channels if double_z else z_channels,
                          kernel_size=3,
                          stride=1,
                          padding_type=padding_type)
        )
        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.res_stack(h)
        h = self.final_block(h)
        return h


class DecoderRe(nn.Module):
    def __init__(self, n_hiddens, upsample, z_channels, image_channel, norm_type='group', padding_type='replicate', n_res_layers=2):
        super().__init__()
        self.conv_first = SamePadConv3d(z_channels, n_hiddens, kernel_size=3, padding_type=padding_type)
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)]
        )
        # n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        # max_us = n_times_upsample.max()
        # in_channels = n_hiddens
        self.conv_blocks = nn.ModuleList()
        for i, step in enumerate(upsample):
            block = nn.Module()
            in_channels = n_hiddens
            out_channels = n_hiddens
            stride = [1, upsample[i], upsample[i]]
            stride = tuple(stride)

            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=stride)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)

        self.conv_out = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = x
        h = self.conv_first(h)
        h = self.res_stack(h)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_out(h)
        return h


# unit test
if __name__ == '__main__':
    encoder = EncoderRe(n_hiddens=320, downsample=[1, 2, 2, 2], z_channels=8, double_z=True, image_channel=96,
                      norm_type='group', padding_type='replicate')
    encoder = encoder.cuda()
    en_input = torch.rand(1, 96, 3, 256, 256).cuda()
    out = encoder(en_input)
    print(out.shape)
    mean, logvar = torch.chunk(out, 2, dim=1)
    # print(mean.shape)
    decoder = DecoderRe(n_hiddens=320, upsample=[2, 2, 2, 1], z_channels=8,   image_channel=96,
                      norm_type='group' )

    decoder = decoder.cuda()
    out = decoder(mean)
    print(out.shape)
    # logvar = nn.Parameter(torch.ones(size=()) * 0.0)
    # print(logvar)
