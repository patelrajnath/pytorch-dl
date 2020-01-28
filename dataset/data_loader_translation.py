#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 10:30 
Date: January 25, 2020	
"""
import os
from collections import Counter

import torch
import tqdm
from torch.utils.data import Dataset


class TranslationDataSet(Dataset):
    """
    """
    def __init__(self, corpus_path_prefix, src, tgt, vocab_src, vocab_tgt, max_size,
                 corpus_lines=None, encoding="utf-8", on_memory=True):
        self.corpus_path_prefix = corpus_path_prefix
        self.corpus_lines = corpus_lines
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.src = src
        self.tgt = tgt
        self.max_size = max_size
        self.lines_src = list()
        self.lines_tgt = list()

        for lang in (self.src, self.tgt):
            with open(self.corpus_path_prefix + '.' + lang, "r", encoding=encoding) as f:
                if lang == self.src:
                    self.lines_src = [line[:-1].strip()
                             for line in tqdm.tqdm(f, desc="Loading {} Dataset...".format(lang),
                                                   total=self.corpus_lines)]
                else:
                    self.lines_tgt = [line[:-1].strip()
                                      for line in tqdm.tqdm(f, desc="Loading {} Dataset...".format(lang),
                                                            total=self.corpus_lines)]
        self.corpus_lines = len(self.lines_tgt)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        line_src = self.get_corpus_line(idx, self.src)
        line_tgt = self.get_corpus_line(idx, self.tgt)
        src_tokens = self.string_to_index(line_src, vocab=self.vocab_src)
        tgt_tokens = self.string_to_index(line_tgt, vocab=self.vocab_tgt)

        src_tokens = [self.vocab_src.sos_index] + src_tokens + [self.vocab_src.eos_index]
        tgt_tokens = [self.vocab_tgt.sos_index] + tgt_tokens + [self.vocab_tgt.eos_index]

        src_tokens = src_tokens[:self.max_size]
        tgt_tokens = tgt_tokens[:self.max_size]

        padding = [self.vocab_src.pad_index for _ in range(self.max_size - len(src_tokens))]
        src_tokens.extend(padding)
        padding = [self.vocab_tgt.pad_index for _ in range(self.max_size - len(tgt_tokens))]
        tgt_tokens.extend(padding)

        output = {
            "source": src_tokens,
            "target": tgt_tokens
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def string_to_index(self, sentence, vocab):
        tokens = sentence.split()
        for i, token in enumerate(tokens):
            tokens[i] = vocab.stoi.get(token, vocab.unk_index)
            if tokens[i] == vocab.unk_index:
                print(token, 'is unk')
        return tokens

    def get_corpus_line(self, index, lang):
        if lang == self.src:
            return self.lines_src[index]
        else:
            return self.lines_tgt[index]
