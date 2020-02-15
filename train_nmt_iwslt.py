#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 15:47 
Date: February 15, 2020	
"""
import math
import os
from math import inf
import time

import torch
import tqdm
from torch import nn

from dataset.iwslt_data import get_data, MyIterator, batch_size_fn, rebatch, SimpleLossCompute, LabelSmoothing

from dataset.iwslt_data import NoamOpt
from models.transformer import TransformerEncoderDecoder
from models.utils.model_utils import save_state
from optim.lr_warm_up import GradualWarmupScheduler

train, val, test, SRC, TGT = get_data()

pad_idx = TGT.vocab.stoi["<blank>"]

BATCH_SIZE = 1000
model_dim=32
heads=2
depth=1

n_batches = math.ceil(len(train) / BATCH_SIZE)

train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)

model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=0.1, depth=depth, num_emb=len(SRC.vocab),
                                  num_emb_target=len(TGT.vocab), max_len=80,
                                  mask_future_steps=True)

# Initialize parameters with Glorot / fan_avg.
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
# optimizer = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
optimizer = NoamOpt(model_dim, depth, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=1000,
#                                           after_scheduler=scheduler_cosine)
compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_condition else "cpu")

if cuda_condition:
    model.cuda()

if cuda_condition and torch.cuda.device_count() > 1:
    print("Using %d GPUS for BERT" % torch.cuda.device_count())
    model = nn.DataParallel(model, device_ids=[0,1,2,3])


def truncate_division(x, y):
    return round(x/y, 2)


modeldir = "transformer"
previous_best = inf
for epoch in range(1, 20):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    print(epoch, n_batches)
    i = 0
    for batch in train_iter:
        print(epoch, i)
        i += 1
        bs = batch.batch_size
        src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
        print('src: ', src)
        print('trg: ', trg)
        src_len = src.size()[1]
        tgt_len = trg.size()[1]
        src_lengths = src.new(bs).fill_(src_len)
        tgt_lengths = src.new(bs).fill_(tgt_len)
        # print(bs, src_lengths.shape, tgt_lengths.shape)
        # exit(0)
        out = model(src, src_lengths, trg, tgt_lengths)
        loss = compute_loss(out, trg,  tgt_lengths.sum())
        total_loss += loss
        total_tokens += tgt_lengths.sum()
        tokens += tgt_lengths.sum()

    #     if i % 50 == 0 and i > 0:
    #         elapsed = time.time() - start
    #         print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
    #               (epoch, i, loss / batch.ntokens, tokens / elapsed))
    #         start = time.time()
    #         tokens = 0
    #         try:
    #             os.makedirs(modeldir)
    #         except OSError:
    #             pass
    #         checkpoint = "checkpoint.{}.".format(truncate_division(total_loss, i)) + 'epoch' + str(epoch) + ".pt"
    #         save_state(os.path.join(modeldir, checkpoint), model, criterion, optimizer, epoch)
    # try:
    #     os.makedirs(modeldir)
    # except OSError:
    #     pass
    # loss_average = truncate_division(total_loss, len(train_iter))
    # checkpoint = "checkpoint.{}.".format(loss_average) + 'epoch' + str(epoch) + ".pt"
    # save_state(os.path.join(modeldir, checkpoint), model, criterion, optimizer, epoch)
    # print('Average loss: {}'.format(total_loss / len(train_iter)))
    # print('PPL: {}'.format(math.exp(total_loss / len(train_iter))))
    #
    # if previous_best > loss_average :
    #     save_state(os.path.join(modeldir, 'checkpoints_best.pt'), model, criterion, optimizer, epoch)
    #     previous_best = loss_average
