#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""


import torch
import numpy as np

from models.transformer import SelfAttention, TransformerBlock, TransformerEncoder, TransformerEncoderDecoder

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05 
Date: January 18, 2020	
"""

import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.data_loader import BertDataSet
from dataset.vocab import WordVocab
from models.bert import Bert
from models.bert_lm import BertLanguageModel

with open("experiments/sample-data/bert-example.txt") as f:
    vocab = WordVocab(f)
    vocab.save_vocab("experiments/sample-data/vocab.pkl")

vocab = WordVocab.load_vocab("experiments/sample-data/vocab.pkl")

lr_warmup = 1000
batch_size = 16
k=128
h=8
max_size=80
data_set = BertDataSet("experiments/sample-data/bert-example.txt", vocab, max_size)

data_loader = DataLoader(data_set, batch_size=batch_size)
vocab_size = len(vocab.stoi)
model = TransformerEncoderDecoder(k, h, depth=2, num_emb=vocab_size, num_emb_target=vocab_size, max_len=max_size)

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


for _ in range(100):
    avg_loss = 0
    # Setting the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc="Running...",
                          total=len(data_loader))
    for i, data in data_iter:
        data = {key: value.to(device) for key, value in data.items()}
        bert_input, bert_label, segment_label, is_next = data
        mask_out, sentence_pred = model(data[bert_input])

        mask_loss = criterion(mask_out.transpose(1, 2), data[bert_label])
        next_loss = criterion(sentence_pred, data[is_next])
        loss = next_loss + mask_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedular.step()
        avg_loss += loss.item()
    print(avg_loss)