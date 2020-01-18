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
data_set = BertDataSet("experiments/sample-data/bert-example.txt", vocab, max_size=512)

lr_warmup = 1000
batch_size = 4

data_loader = DataLoader(data_set, batch_size=4)

vocab_size = len(vocab.stoi)
k=32
h=8
w=20
b=16

x = np.random.rand(b, w, k)

model = TransformerEncoderDecoder(k, h, depth=2, num_emb=vocab_size, num_emb_target=vocab_size, max_len=80)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for _ in range(10):
    avg_loss = 0
    # Setting the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc="Running...",
                          total=len(data_loader))
    for i, data in data_iter:
        data = {key: value.to(device) for key, value in data.items()}
        bert_input, bert_label, segment_label, is_next = data
        model(data[bert_input])