#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 09:59 
Date: January 25, 2020	
"""
import sys

import sentencepiece as sp

spm = sp.SentencePieceProcessor()
spm.Load('m.model')
input_file = sys.argv[1]
with open(input_file) as inp, \
        open(input_file + '.enc', 'w') as out:
    for line in inp:
        encoded = spm.encode_as_pieces(line)
        out.write(' '.join(encoded) + '\n')

# text_ids = spm.encode_as_ids("Only then can we speak of 'communication'.")
# print(text_ids)
# print(spm.decode_ids(text_ids))
# print(spm.get_piece_size())
