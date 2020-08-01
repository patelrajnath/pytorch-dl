#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 10:30 
Date: January 25, 2020	
"""

import random

import torch
import tqdm
from torch.utils.data import Dataset


class MBartDataSet(Dataset):
    """
    """
    def __init__(self, corpus_path, vocab, max_size, corpus_lines=None, encoding="utf-8", add_sos_eos=True, on_memory=True):
        self.corpus_path = corpus_path
        self.corpus_lines = corpus_lines
        self.vocab = vocab
        self.max_size = max_size
        self.add_sos_eos = add_sos_eos

        with open(self.corpus_path, "r", encoding=encoding) as f:
            self.lines = [line[:-1].strip()
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=self.corpus_lines)]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        line = self.get_corpus_line(idx)
        t1_tokens, t1_label = self.random_words(line)

        if self.add_sos_eos:
            t1_random = [self.vocab.sos_index] + t1_tokens + [self.vocab.eos_index]
            t1_label = [self.vocab.sos_index] + t1_label + [self.vocab.eos_index]
        else:
            t1_random = t1_tokens
            t1_label = t1_label

        input = t1_random[:self.max_size]
        label = t1_label[:self.max_size]

        output = {
            "source": input,
            "target": label
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_words(self, sentence):
        tokens = sentence.split()
        output_labels = list()
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.35:
                prob /= 0.35
                # 80% of the tokens we replace with mask
                if prob < 0.80:
                    tokens[i] = self.vocab.mask_index
                # 10% of tokens to be replaced with random word
                elif prob < 0.90:
                    tokens[i] = random.randrange(len(self.vocab.stoi))
                # Remaining 10% we keep actual word
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

            # add correct tokens to be predicted during training
            output_labels.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        return tokens, output_labels

    def get_corpus_line(self, index):
        return self.lines[index]
