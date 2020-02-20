#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 2:31 PM,  2/20/20
"""
from torch.utils.data.dataloader import DataLoader

from dataset.bilingual_data_iter import MyIterableDataset
from dataset.iwslt_data import rebatch_data
from dataset.vocab import WordVocab
from models.utils.model_utils import my_collate

src = 'en'
trg = 'it'
input_file = 'sample-data/europarl.enc'
for lang in (src, trg):
    with open(input_file + '.' + lang) as f:
        vocab = WordVocab(f)
        vocab.save_vocab("sample-data/{}.pkl".format(lang))

vocab_src = WordVocab.load_vocab("sample-data/{}.pkl".format(src))
vocab_trg = WordVocab.load_vocab("sample-data/{}.pkl".format(trg))

# Only useful in case we don't need shuffling of data
dataset = MyIterableDataset(filename='sample-data/europarl.enc', src=src, trg=trg,
                         vocab_src=vocab_src, vocab_trg=vocab_trg)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=my_collate)
for epoch in range(10):
    for i, batch in enumerate(rebatch_data(batch=b, pad_idx=1, device='cpu') for b in dataloader):
        print(batch.src, batch.trg)

    print(epoch)
