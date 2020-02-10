#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""
from torch import nn

from models.embeddings.bert_embeddings import BertEmbeddings
from models.transformer import TransformerBlock


class Bert(nn.Module):
    """
    """
    def __init__(self, vocab_size, width=128, depth=4 ,heads=8):
        super().__init__()
        self.h = width
        self.d = depth
        self.heads = heads
        self.vocab_size = vocab_size

        self.bert_embeddings = BertEmbeddings(vocab_size, self.h)

        tblocks = []
        for i in range(self.d):
            tblocks.append(TransformerBlock(emb_dim=self.h, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, x, segment_label):
        x = self.bert_embeddings(x, segment_label)
        x = self.tblocks(x)
        return x
