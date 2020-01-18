#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

from torch import nn


class SegmentEmbedding(nn.Module):
    """

    """
    def __init__(self, labels, emb_dim):
        super().__init__()
        self.seg_emb = nn.Embedding(labels, emb_dim)

    def forward(self, x):
        return self.seg_emb(x)