#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 09:59 
Date: January 25, 2020	
"""

import sentencepiece as sp

spm = sp.SentencePieceProcessor()
spm.Load('m.model')
with open('experiments/sample-data/europarl.it') as inp, \
        open('experiments/sample-data/europarl.it.enc', 'w') as out:
    for line in inp:
        encoded = spm.encode_as_pieces(line)
        out.write(' '.join(encoded) + '\n')


# text_ids = spm.encode_as_ids("Only then can we speak of 'communication'.")
# print(text_ids)
# print(spm.decode_ids(text_ids))
# print(spm.get_piece_size())
