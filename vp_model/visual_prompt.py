# The code referenced above is sourced from [Understanding and Improving Visual Prompting: A Label-Mapping Perspective].
# You can find the original repository at the following link:
# [https://github.com/OPTML-Group/ILM-VP/blob/main/models/visual_prompt.py#L7].

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def patch_to_image(x):
    """
    patch representation을 이미지로 복원합니다.

    Args:
        x (torch.Tensor): 입력 텐서, shape (B, 1, N, C, T, ph, pw)
            예) (2, 1, 324, 3, 2, 14, 14)
            여기서
              B: 배치 크기
              1: 추가 차원 (예: 슬라이스 개수)
              N: 공간 패치 개수 (예: 324 = 18x18)
              C: 채널 수 (예: 3)
              T: temporal patch 크기 (예: 2)
              ph, pw: 각 패치의 height, width (예: 14, 14)

    Returns:
        torch.Tensor: 복원된 이미지, shape (B, C, H, W)
            예) (2, 3, 252, 252)  [H = 18*14, W = 18*14]
    """
    # 1. temporal patch 차원(T, dim=4)을 평균 내어 제거 (미분가능)
    x = x.mean(dim=4)  # (B, 1, N, C, ph, pw)

    # 2. 두 번째 차원이 1이므로 squeeze하여 (B, N, C, ph, pw)로 만듭니다.
    x = x.squeeze(1)  # (B, N, C, ph, pw)

    B, N, C, ph, pw = x.shape
    grid_size = int(math.sqrt(N))  # N가 완전제곱수라고 가정 (예: 18)

    # 3. 1차원 패치 차원(N)을 2차원 그리드로 reshape: (B, grid_size, grid_size, C, ph, pw)
    x = x.view(B, grid_size, grid_size, C, ph, pw)

    # 4. 차원 순서를 (B, C, grid_size, ph, grid_size, pw)로 변경
    x = x.permute(0, 3, 1, 4, 2, 5)

    # 5. grid와 패치 차원을 합쳐서 최종 이미지 (B, C, grid_size*ph, grid_size*pw)
    x = x.reshape(B, C, grid_size * ph, grid_size * pw)
    return x

def image_to_patch_qwen(x, patch_size=14, temporal_size=2):
    """
    이미지를 패치 representation으로 분해합니다.

    Args:
        x (torch.Tensor): 입력 이미지, shape (B, C, H, W)
            예) (2, 3, 252, 252)
            여기서 H와 W는 patch_size의 정수배여야 합니다.
        patch_size (int): 각 패치의 height와 width (기본값 14)
        temporal_size (int): temporal patch 차원의 크기 (복제를 위해, 기본값 2)

    Returns:
        torch.Tensor: 패치 representation, shape (B, 1, N, C, T, patch_size, patch_size)
            예) (2, 1, 324, 3, 2, 14, 14)  (여기서 N = (H/patch_size)^2, T = temporal_size)
    """
    B, C, H, W = x.shape
    grid_size = H // patch_size  # H와 W가 patch_size의 정수배라고 가정 (예: 252/14 = 18)
    N = grid_size * grid_size  # 총 패치 수 (예: 18*18 = 324)

    # 1. 이미지를 공간 패치로 분리
    #    먼저 (B, C, grid_size, patch_size, grid_size, patch_size)로 reshape
    x = x.view(B, C, grid_size, patch_size, grid_size, patch_size)

    # 2. 패치들을 grid 형태로 배치: (B, grid_size, grid_size, C, patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)

    # 3. grid를 하나의 차원으로 합쳐 (B, N, C, patch_size, patch_size)
    x = x.reshape(B, N, C, patch_size, patch_size)

    # 4. 원래 representation에 맞게 차원 추가:
    #    (B, 1, N, C, patch_size, patch_size)
    x = x.unsqueeze(1)

    # 5. temporal patch 차원(T)을 추가합니다. (B, 1, N, C, 1, patch_size, patch_size)
    x = x.unsqueeze(4)

    # 6. temporal 차원을 temporal_size만큼 복제하여 (B, 1, N, C, T, patch_size, patch_size)로 만듭니다.
    x = x.repeat(1, 1, 1, 1, temporal_size, 1, 1)

    return x

class ExpansiveVisualPrompt(nn.Module):
    def __init__(self, pad_size, target_size, model_name, init='randn', normalize=None):
        super(ExpansiveVisualPrompt, self).__init__()
        self.pad_size = pad_size
        self.target_size = target_size
        self.model_name = model_name

        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(3, pad_size, pad_size))
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(3, pad_size, pad_size))
        else:
            raise ValueError("init method not supported")
        self.normalize = normalize

        self.l_pad = int((pad_size-target_size+1)/2)
        self.r_pad = int((pad_size-target_size)/2)

        mask = torch.zeros(3, target_size, target_size)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))

    def forward(self, x):
        if 'Llama' in self.model_name:
            b, t, d, c, h, w = x.shape
            x = x.view(b * t * d, c, h, w)
        elif 'Qwen' in self.model_name:
            x = x.view(-1, 1, 324, 3, 2, 14, 14)
            x = patch_to_image(x)

        x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask

        if self.normalize is not None:
            x = self.normalize(x)

        if 'Llama' in self.model_name:
            x = x.view(b, t, d, c, self.pad_size, self.pad_size)
        elif 'Qwen' in self.model_name:
            x = image_to_patch_qwen(x)
            x = x.view(-1, 3, 2, 14, 14)
        return x