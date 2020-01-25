#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

from torch import nn

from models.embeddings.position_emb import PositionEmbedding
from models.embeddings.seg_emb import SegmentEmbedding
from models.embeddings.token_emb import TokenEmbedding


class BertEmbeddings(nn.Module):
    """
    """
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, emb_dim)
        self.position_emb = PositionEmbedding(emb_dim)
        self.segment_emb = SegmentEmbedding(3, emb_dim)
        self.dropout = nn.Dropout(0.01)
        self.embedding_dim = emb_dim

    def forward(self, x, segment_label):
        x = self.token_emb(x) + self.position_emb(x) + self.segment_emb(segment_label)
        return self.dropout(x)