#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""
import os
import sys

from models.transformer import TransformerEncoderDecoder

import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.data_loader_mbert import MBertDataSet
from dataset.vocab import WordVocab
from models.utils.model_utils import save_state

input_file = sys.argv[1]
with open(input_file) as f:
    vocab = WordVocab(f)
    vocab.save_vocab("experiments/sample-data/vocab.pkl")

vocab = WordVocab.load_vocab("experiments/sample-data/vocab.pkl")

lr_warmup = 500
batch_size = int(sys.argv[2])
k=512
h=int(sys.argv[3])
depth=int(sys.argv[4])
max_size=80
modeldir = "bert"
data_set = MBertDataSet(input_file, vocab, max_size)

data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
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
        if i % 1000 == 0 and i > 0:
            checkpoint = "checkpoint.{}.".format(avg_loss/i) + str(epoch) + ".pt"
            try:
                os.makedirs(modeldir)
            except OSError:
                pass
            save_state(os.path.join(modeldir, checkpoint), model, criterion, optimizer, epoch)
    print('Average loss: {}'.format(avg_loss / len(data_iter)))