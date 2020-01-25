#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""
from models.transformer import TransformerEncoderDecoder

import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.data_loader_mbert import MBertDataSet
from dataset.vocab import WordVocab

with open("experiments/sample-data/europarl.en.enc") as f:
    vocab = WordVocab(f)
    vocab.save_vocab("experiments/sample-data/vocab.pkl")

vocab = WordVocab.load_vocab("experiments/sample-data/vocab.pkl")

lr_warmup = 500
batch_size = 16
k=512
h=4
depth=1
max_size=80
data_set = MBertDataSet("experiments/sample-data/europarl.en.enc", vocab, max_size)

data_loader = DataLoader(data_set, batch_size=batch_size)
vocab_size = len(vocab.stoi)
model = TransformerEncoderDecoder(k, h, depth=depth, num_emb=vocab_size, num_emb_target=vocab_size, max_len=max_size)

criterion = nn.NLLLoss(ignore_index=0)
optimizer = Adam(lr=0.0001, params=model.parameters())
lr_schedular = lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0))

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_condition else "cpu")

if cuda_condition:
    model.cuda()

if cuda_condition and torch.cuda.device_count() > 1:
    print("Using %d GPUS for BERT" % torch.cuda.device_count())
    model = nn.DataParallel(model, device_ids=[0,1,2,3])

for epoch in range(100):
    avg_loss = 0
    # Setting the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc="Running epoch: {}".format(epoch),
                          total=len(data_loader))
    for i, data in data_iter:
        data = {key: value.to(device) for key, value in data.items()}
        bert_input, bert_label = data
        mask_out = model(data[bert_input])
        loss = criterion(mask_out.transpose(1, 2), data[bert_label])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedular.step(epoch)
        avg_loss += loss.item()
    print(avg_loss / len(data_iter))