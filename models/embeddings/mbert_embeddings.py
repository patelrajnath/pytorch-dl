#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:01 
Date: January 25, 2020	
"""
from torch import nn

from models.embeddings.position_emb import PositionEmbedding
from models.embeddings.token_emb import TokenEmbedding


class MBertEmbeddings(nn.Module):
    """
    """
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, emb_dim)
        self.position_emb = PositionEmbedding(emb_dim)
        self.dropout = nn.Dropout(0.01)
        self.embedding_dim = emb_dim

    def forward(self, x):
        x = self.token_emb(x) + self.position_emb(x)
        return self.dropout(x)