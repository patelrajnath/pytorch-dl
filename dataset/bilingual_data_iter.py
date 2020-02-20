#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 09:11 
Date: February 20, 2020	
"""
import torch
from torch.utils.data.dataset import IterableDataset


class MyIterableDataset(IterableDataset):

    def __init__(self, filename, src, trg, vocab_src, vocab_trg):
        # Store the filename in object's memory
        self.filename = filename
        self.src = src
        self.trg = trg
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        # And that's it, we no longer need to store the contents in the memory

    def preprocess(self, text_src, text_trg):
        src_tokens = self.string_to_index(text_src, vocab=self.vocab_src)
        tgt_tokens = self.string_to_index(text_trg, vocab=self.vocab_trg)
        return src_tokens, tgt_tokens

    def line_mapper(self, line_src, line_trg):
        # Splits the line into text and label and applies preprocessing to the text
        # text, label = line.split(',')
        text_src, text_trg = self.preprocess(line_src, line_trg)
        data = {
            "source": text_src,
            "target": text_trg
        }
        return {key: torch.tensor(value) for key, value in data.items()}

    def __iter__(self):
        # Create an iterator
        file_itr_src = open(self.filename + '.' + self.src)
        file_itr_trg = open(self.filename + '.' + self.trg)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr_src, file_itr_trg)

        return mapped_itr

    def string_to_index(self, sentence, vocab):
        tokens = sentence.split()
        for i, token in enumerate(tokens):
            tokens[i] = vocab.stoi.get(token, vocab.unk_index)
            if tokens[i] == vocab.unk_index:
                print(token, 'is unk')
        return tokens
