# The code referenced above is sourced from [Understanding and Improving Visual Prompting: A Label-Mapping Perspective].
# You can find the original repository at the following link:
# [https://github.com/OPTML-Group/ILM-VP/blob/main/models/visual_prompt.py#L7].

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExpansiveVisualPrompt(nn.Module):
    def __init__(self, pad_size, target_size, init = 'randn', normalize=None):
        super(ExpansiveVisualPrompt, self).__init__()
        self.pad_size = pad_size
        self.target_size = target_size
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
        b, t, d, c, h, w = x.shape
        x = x.view(b * t * d, c, h, w)  # Reshape to [new_batch, channel, height, width]
        x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask

        if self.normalize is not None:
            x = self.normalize(x)

        x = x.view(b, t, d, c, self.pad_size, self.pad_size)
        return x