from math import sqrt

import torch
import torch.nn as nn

from transformers import MllamaForConditionalGeneration, AutoProcessor
from models.layers.Embed import PatchEmbedding1D

class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.target_window = target_window

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window * 34)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], self.target_window, -1)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_len = config['pred_len']
        self.image_size = config['image_size']
        self.d_llm = config['d_llm']
        self.d_ff = config['d_ff']

        if config['llm_model'] == 'Llama-3.2-11B-Vision':
            self.vlm_model = MllamaForConditionalGeneration.from_pretrained(
                'meta-llama/Llama-3.2-11B-Vision',
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision')
        for param in self.vlm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(config['dropout'])

        self.output_projection = FlattenHead(self.head_nf, self.pred_len,
                                             head_dropout=config["dropout"])

