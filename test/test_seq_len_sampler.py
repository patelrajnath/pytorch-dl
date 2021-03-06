#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 20:01 
Date: February 19, 2020	
"""
from torch.utils.data.dataloader import DataLoader

from dataset.data_loader_translation import TranslationDataSet, BySequenceLengthSampler
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
vocab_tgt = WordVocab.load_vocab("sample-data/{}.pkl".format(trg))

data_set = TranslationDataSet(input_file, src, trg, vocab_src, vocab_tgt, 100,
                              add_sos_and_eos=True)

bucket_boundaries = [i*10 for i in range(30)]
batch_sizes = 10

sampler = BySequenceLengthSampler(data_set, bucket_boundaries, batch_sizes)

data_loader = DataLoader(data_set, batch_sampler=sampler, collate_fn=my_collate)
# data_loader = DataLoader(data_set, batch_size=batch_sizes, collate_fn=my_collate)

for i, batch in enumerate(data_loader):
    print(batch)
    if i> 100:
        break
