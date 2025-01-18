# The code referenced above is sourced from [Understanding and Improving Visual Prompting: A Label-Mapping Perspective].
# You can find the original repository at the following link:
# [https://github.com/OPTML-Group/ILM-VP/blob/main/models/visual_prompt.py#L7].

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExpansiveVisualPrompt(nn.Module):
    def __init__(self, out_size, mask, init = 'zero', normalize=None):
        super(ExpansiveVisualPrompt, self).__init__()
        assert mask.shape[0] == mask.shape[1]
        in_size = mask.shape[0]
        self.out_size = out_size
        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(3, out_size, out_size))
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(3, out_size, out_size))
        else:
            raise ValueError("init method not supported")
        self.normalize = normalize

        self.l_pad = int((out_size-in_size+1)/2)
        self.r_pad = int((out_size-in_size)/2)

        mask = np.repeat(np.expand_dims(mask, 0), repeats=3, axis=0)
        mask = torch.Tensor(mask)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x