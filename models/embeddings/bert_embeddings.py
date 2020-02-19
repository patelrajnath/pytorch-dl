#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

from torch import nn

from models.embeddings.position_emb import PositionalEncoding
from models.embeddings.seg_emb import SegmentEmbedding
from models.embeddings.token_emb import Embeddings


class BertEmbeddings(nn.Module):
    """
    """
    def __init__(self, vocab_size, emb_dim, dropout=0.1):
        super().__init__()
        self.token_emb = Embeddings(emb_dim, vocab_size)
        self.position_emb = PositionalEncoding(emb_dim, dropout)
        self.segment_emb = SegmentEmbedding(3, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = emb_dim

    def forward(self, x, segment_label):
        x = self.token_emb(x)

        # position embedding adds the token embedding and returns the final embedding
        x = self.position_emb(x)

        x += self.segment_emb(segment_label)

        return self.dropout(x)