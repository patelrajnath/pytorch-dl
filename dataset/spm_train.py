#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 10:00 
Date: January 25, 2020	
"""
import sys

import sentencepiece as spm
input_file = sys.argv[1]
input_sentnce_size = sys.argv[2]
spm.SentencePieceTrainer.Train('--input={} --input_sentence_size={} --shuffle_input_sentence=true '
                               '--model_prefix=m --vocab_size={}'.format(input_file, input_sentnce_size, 100000))