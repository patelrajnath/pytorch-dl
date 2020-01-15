#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 3:53 PM,  1/15/20
"""
from collections import Counter


class TorchVocab(object):
    def __init__(self, counter, max_size, min_freq):
        self.counter = counter
        self.min_freq = min_freq
        self.max_len = max_size


class Vocab(TorchVocab):
    def __index__(self, counter, max_size, min_freq):
        super().__init__(counter, max_size, min_freq)

    def to_seq(self):
        pass

    def from_seq(self):
        pass


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=None):
        counter = Counter()
        for text in texts:
            if isinstance(text, list):
                words = text
            else:
                words = text.strip().split()

            for word in words:
                counter[word] += 1

        super().__init__(counter=counter, max_size=max_size, min_freq=min_freq)


with open("experiments/data/bert-example.txt") as f:
    vocab = WordVocab(f)
    print(vocab.counter)