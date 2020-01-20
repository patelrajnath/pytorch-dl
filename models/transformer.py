#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.embeddings.bert_embeddings import BertEmbeddings
from models.utils.model_utils import mask_, d


class SelfAttention(nn.Module):
    """
    This is basic transformer model
    """

    def __init__(self, k, heads, mask_future_steps=False):
        super().__init__()
        self.k, self.heads, self.mask_future_steps = k, heads, mask_future_steps
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalue = nn.Linear(k, k * heads, bias=False)
        self.tokey = nn.Linear(k, k * heads, bias=False)
        self.unifyheads = nn.Linear(k * heads, k, bias=False)

    def forward(self, x, enc=None):
        b, t, k = x.size()

        if type(enc) != type(None):
            enc = enc
        else:
            enc = x

        h = self.heads
        query = self.toqueries(x).view(b, t, h, k)
        key = self.tokey(enc).view(b, t, h, k)
        value = self.tovalue(enc).view(b, t, h, k)

        query = query.transpose(1, 2).contiguous().view(b * h, t, k)
        key = key.transpose(1, 2).contiguous().view(b * h, t, k)
        value = value.transpose(1, 2).contiguous().view(b * h, t, k)

        query = query / (k ** (1 / 4))
        key = key / (k ** (1 / 4))

        dot = torch.bmm(query, key.transpose(1, 2))
        if self.mask_future_steps:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # print(torch.sum(dot, dim=2))
        out = torch.bmm(dot, value).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)
        # print('key:', key.size(), 'value:', value.size(), 'query:', query.size(),
        #       'dot', dot.size(), "out", out.size())


class TransformerBlock(nn.Module):
    def __init__(self, k, heads, ff=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, ff * k),
            nn.ReLU(),
            nn.Linear(ff * k, k))

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        ff_out = self.ff(x)
        x = self.norm2(ff_out + x)
        x = self.do(x)
        return x


class TransformerBlockDecoder(nn.Module):
    def __init__(self, k, heads, ff=4, dropout=0.0, mask_future_steps=False):
        super().__init__()

        # Masked self attention
        self.attention = SelfAttention(k, heads=heads, mask_future_steps=mask_future_steps)

        # Encoder-decoder self attention
        self.attention_encoder_decoder = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.norm3 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, ff * k),
            nn.ReLU(),
            nn.Linear(ff * k, k))

        self.linear = nn.Linear(k, k)

        self.do = nn.Dropout(dropout)

    def forward(self, x, enc):
        masked_attention = self.attention(x)
        # Add and layer normalize
        x = self.norm1(masked_attention + x)

        encdec_attention = self.attention_encoder_decoder(x, enc)

        # Add and layer normalize
        x = self.norm2(encdec_attention + x)
        x = self.do(x)

        # Run feed-forward
        ff_out = self.ff(x)

        # Add and layer normalize
        x = self.norm3(ff_out + x)
        x = self.do(x)
        return x


class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes, dropout=0.0):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t, device=d())
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class
        # probabilities
        x = self.toprobs(x.mean(dim=1))

        return F.log_softmax(x, dim=1)


class TransformerEncoder(nn.Module):
    def __init__(self, k, heads, depth, num_emb, max_len):
        super().__init__()
        self.max_len = max_len
        self.bert_embeddings = BertEmbeddings(num_emb, k)


        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(k, heads))
        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, x, segment_label):
        bert_emb = self.bert_embeddings(x, segment_label)
        encoding = self.tblocks(bert_emb)
        return encoding


class TransformerDecoder(nn.Module):
    def __init__(self, k, heads, depth, num_emb_target, max_len, mask_future_steps=False):
        super().__init__()
        self.bert_emb = BertEmbeddings(num_emb_target, k)
        self.max_len = max_len

        self.tblocks_decoder = nn.ModuleList()
        for _ in range(depth):
            self.tblocks_decoder.append(TransformerBlockDecoder(k, heads, mask_future_steps))

    def forward(self, x, segment_label, enc):
        bert_emb = self.bert_emb(x, segment_label)
        inner_state = [bert_emb]
        for i, layer in enumerate(self.tblocks_decoder):
            bert_emb = layer(bert_emb, enc, mask_future_steps=True)
            inner_state.append(x)
        return bert_emb


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, k, heads, depth, num_emb, num_emb_target, max_len, c=3, mask_future_steps=True):
        super().__init__()
        self.encoder = TransformerEncoder(k, heads, depth, num_emb, max_len)
        self.decoder = TransformerDecoder(k, heads, depth, num_emb_target, max_len, mask_future_steps)
        self.ff = nn.Linear(k, num_emb_target)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.ff_next_sentence = nn.Linear(k, c)
        self.softmax_next_sentence = nn.LogSoftmax(dim=-1)

    def forward(self, x, segment_label):
        enc = self.encoder(x, segment_label)
        enc_dec = self.decoder(x, segment_label, enc)
        ff_out = self.ff(enc_dec)
        ff_next_sentence_out = self.ff_next_sentence(enc_dec)
        return self.softmax(ff_out), self.softmax_next_sentence(ff_next_sentence_out[:, 0])
