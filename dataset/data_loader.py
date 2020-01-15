#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 2:11 PM,  1/15/20
"""
import random
import tqdm
from torch.utils.data import Dataset, DataLoader


class BertDataSet(Dataset):
    """
    """
    def __init__(self, corpus_path, corpus_lines=None, encoding="utf-8", on_memory=True):
        self.corpus_path = corpus_path
        self.corpus_lines = corpus_lines

        with open(self.corpus_path, "r", encoding=encoding) as f:
            self.lines = [line[:-1].split("\t")
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=self.corpus_lines)]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        t1, t2, is_next = self.random_sent(idx)
        self.random_words(t1)
        self.random_words(t2)
        return t1, t2, is_next

    def random_words(self, sentence):
        tokens = sentence.split()
        for i, token in enumerate(tokens):
            i, token = i, token
        return None

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        prob = random.random()
        if prob >= 0.50:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, index):
        return self.lines[index][0], self.lines[index][1]

    def get_random_line(self):
        return self.lines[random.randrange(self.corpus_lines)][1]


data_set = BertDataSet("experiments/data/bert-example.txt")
data_loader = DataLoader(data_set, batch_size=2)
for batch in data_loader:
    print(batch)