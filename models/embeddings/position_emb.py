import math

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    """
    """
    def __init__(self, emb_dim, seq_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(seq_len, emb_dim).float()
        pe.require_grad = False

        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = (torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]