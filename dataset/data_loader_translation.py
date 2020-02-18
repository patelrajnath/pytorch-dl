#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 10:30 
Date: January 25, 2020	
"""

import torch
import tqdm
from torch.utils.data import Dataset

import numpy as np
from random import shuffle
from torch.utils.data.sampler import Sampler


class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source,
                 bucket_boundaries, batch_size=64, ):
        super().__init__(data_source)
        self.data_source = data_source
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append((i, p.shape[0]))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                                         , int(data_buckets[k].shape[0] / self.batch_size)))
        shuffle(iter_list)  # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return len(self.data_source)

    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less_equal(buckets_min, seq_length),
            np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


class TranslationDataSet(Dataset):
    """
    """
    def __init__(self, corpus_path_prefix, src, tgt, vocab_src, vocab_tgt, max_size,
                 corpus_lines=None, encoding="utf-8", add_sos_and_eos=False, on_memory=True):
        self.corpus_path_prefix = corpus_path_prefix
        self.corpus_lines = corpus_lines
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.src = src
        self.tgt = tgt
        self.add_sos_and_eos = add_sos_and_eos
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

        if self.add_sos_and_eos:
            src_tokens = [self.vocab_src.sos_index] + src_tokens + [self.vocab_src.eos_index]
            tgt_tokens = [self.vocab_tgt.sos_index] + tgt_tokens + [self.vocab_tgt.eos_index]

        if self.max_size:
            src_tokens = src_tokens[:self.max_size]
            tgt_tokens = tgt_tokens[:self.max_size]

        output = {
            "source": src_tokens,
            "target": tgt_tokens,
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
