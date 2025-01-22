import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class ReplicationPad2D(nn.Module):
    def __init__(self, padding):
        """
        Args:
            padding (int or tuple): 패딩 크기. 정수 하나일 경우 모든 방향에 동일한 패딩을 적용하고,
                                     4개의 정수로 된 튜플일 경우 (좌, 우, 상, 하) 순서로 패딩을 적용합니다.
        """
        super(ReplicationPad2D, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple) and len(padding) == 4:
            self.padding = padding
        else:
            raise ValueError("Padding must be an int or a 4-tuple")

    def forward(self, input: Tensor) -> Tensor:
        pad_left, pad_right, pad_top, pad_bottom = self.padding
        batch, channels, height, width = input.size()

        # 좌측 패딩
        if pad_left > 0:
            left_pad = input[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, pad_left)
            input = torch.cat([left_pad, input], dim=-1)

        # 우측 패딩
        if pad_right > 0:
            right_pad = input[:, :, :, -1].unsqueeze(-1).repeat(1, 1, 1, pad_right)
            input = torch.cat([input, right_pad], dim=-1)

        # 상단 패딩
        if pad_top > 0:
            top_pad = input[:, :, 0, :].unsqueeze(2).repeat(1, 1, pad_top, 1)
            input = torch.cat([top_pad, input], dim=2)

        # 하단 패딩
        if pad_bottom > 0:
            bottom_pad = input[:, :, -1, :].unsqueeze(2).repeat(1, 1, pad_bottom, 1)
            input = torch.cat([input, bottom_pad], dim=2)

        return input

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in ** 2, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = self.tokenConv(x)
        x = x.permute(0, 2, 1)
        return x

class PatchEmbedding2D(nn.Module):
    def __init__(self, d_model, patch_size, stride, dropout):
        super(PatchEmbedding2D, self).__init__()
        # Patching
        self.patch_size = patch_size
        self.stride = stride
        self.padding_patch_layer = ReplicationPad2D((stride, stride, stride, stride))
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_size, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1] ** 2
        x = self.padding_patch_layer(x)
        patches = self.unfold(x)  # (배치, 채널 * patch_size * patch_size, num_patches)
        patches = patches.transpose(1, 2)  # (배치, num_patches, 채널 * patch_size * patch_size)
        patches = patches.reshape(patches.size(0), -1, self.patch_size, self.patch_size)

        # Input encoding
        patches = self.value_embedding(patches)
        return self.dropout(patches), n_vars