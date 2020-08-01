#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:01 
Date: January 25, 2020	
"""
from torch import nn

from models.embeddings.position_emb import PositionalEncoding
from models.embeddings.token_emb import Embeddings


class MBartEmbeddings(nn.Module):
    """
    """
    def __init__(self, vocab_size, emb_dim, dropout=0.1):
        super().__init__()
        self.token_emb = Embeddings(vocab_size, emb_dim)
        self.position_emb = PositionalEncoding(emb_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = emb_dim

    def forward(self, x):
        x = self.token_emb(x)
        x = self.position_emb(x)
        return self.dropout(x)
