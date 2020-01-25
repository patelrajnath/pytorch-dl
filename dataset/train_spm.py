#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 10:00 
Date: January 25, 2020	
"""
import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=experiments/sample-data/europarl.en-it '
                               '--model_prefix=m --vocab_size=5000')