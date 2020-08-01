#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

from torch import nn
from models.bert import Bert


class BertLanguageModel(nn.Module):
    """
    """
    def __init__(self, bert: Bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.masked_lm = MaskedLanguageModel(bert.h, vocab_size)
        self.next_sentence = NextSentencePrediction(bert.h)

    def forward(self, x, segment=None):
        x = self.bert(x, segment)
        if segment:
            return self.masked_lm(x), self.next_sentence(x)
        else:
            return self.masked_lm(x)


class MaskedLanguageModel(nn.Module):
    """
    """
    def __init__(self, h, vocab_size):
        super().__init__()
        self.ff = nn.Linear(h, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.ff(x))


class NextSentencePrediction(nn.Module):
    """
    """
    def __init__(self, h, c=2):
        super().__init__()
        self.ff = nn.Linear(h, c)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.ff(x[:, 0]))
