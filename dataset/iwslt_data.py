#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
 Copyright (c) 2018 Raj Nath Patel
 Licensed under the GNU Public License.
 Author: Raj Nath Patel
 Email: patelrajnath (at) gmail (dot) com
 Created: 25/May/2018 12:27
 """

# For utils loading.
from os import path

import dill
import torch
from pathlib import Path
from torch import nn
from torch.autograd import Variable
from torchtext import data, datasets
import numpy as np
from torchtext.data import Dataset


def get_data(args):
    MAX_LEN = 100
    if path.exists("{0}/SRC.fields".format(args.model)) and \
            path.exists("{0}/TGT.fields".format(args.model)):
        print('Fields exists..')

        with open("{0}/SRC.fields".format(args.model), "rb") as f:
            SRC = dill.load(f)

        with open("{0}/TGT.fields".format(args.model), "rb") as f:
            TGT = dill.load(f)

        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                  len(vars(x)['trg']) <= MAX_LEN)
        return train, val, test, SRC, TGT

    else:
        import spacy

        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')

        print('Fields donot exists..')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        BOS_WORD = '<sos>'
        EOS_WORD = '<eos>'
        BLANK_WORD = "<blank>"

        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                         eos_token = EOS_WORD, pad_token=BLANK_WORD)

        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                  len(vars(x)['trg']) <= MAX_LEN)

        MIN_FREQ = 2
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

        with open("{0}/SRC.fields".format(args.model), "wb") as f:
            dill.dump(SRC, f)
        with open("{0}/TGT.fields".format(args.model), "wb") as f:
            dill.dump(TGT, f)

        return train, val, test, SRC, TGT


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of utils with mask during training."

    def __init__(self, src, trg=None, pad=0, src_len=None, trg_len=None, device='cpu'):
        self.src = src.to(device)
        if src_len is not None:
            self.src_len = src_len.to(device)
        if trg_len is not None:
            self.trg_len = trg_len.to(device)

        self.src_mask = (src != pad).unsqueeze(-2).to(device)
        if trg is not None:
            # Reduce the <eos> symbol
            self.trg = trg[:, :-1].to(device)
            # Reduce the <sos> symbol
            self.trg_y = trg[:, 1:].to(device)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad).to(device)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class BatchMBert:
    "Object for holding a batch of utils with mask during training."

    def __init__(self, src, trg=None, pad=0, src_len=None, trg_len=None, device='cpu'):
        self.src = src.to(device)
        if src_len is not None:
            self.src_len = src_len.to(device)
        if trg_len is not None:
            self.trg_len = trg_len.to(device)

        self.src_mask = (src != pad).unsqueeze(-2).to(device)
        if trg is not None:
            # Reduce the <eos> symbol
            self.trg = src[:, :-1].to(device) # For mBert source is used as target
            # Reduce the <sos> symbol
            self.trg_y = trg[:, 1:].to(device)
            self.trg_mask = \
                self.make_std_mask(self.trg_y, pad).to(device)
            self.ntokens = (self.trg != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def rebatch(pad_idx, batch, device='cpu'):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx, device=device)


def rebatch_onmt(pad_idx, batch, device='cpu'):
    "Fix order in torchtext to match ours"
    src, trg = batch.src[0].squeeze(-1).transpose(0, 1), batch.tgt.squeeze(-1).transpose(0, 1)
    return Batch(src, trg, pad_idx, device=device)


def rebatch_data(pad_idx, batch, device='cpu'):
    "Fix order in torchtext to match ours"
    source, targets, lengths_source, lengths_target = batch
    return Batch(source, targets, pad_idx, src_len=lengths_source, trg_len=lengths_target, device=device)


def rebatch_mbert(pad_idx, batch, device='cpu'):
    "Fix order in torchtext to match ours"
    source, targets, lengths_source, lengths_target = batch
    return BatchMBert(source, targets, pad_idx, src_len=lengths_source, trg_len=lengths_target, device=device)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x, _ = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def save_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path/"examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path/"fields.pkl", pickle_module=dill)


def load_dataset(path):
    if not isinstance(path, Path):
        path = Path(path)
    print(path/"examples.pkl")
    examples = torch.load(path/"examples.pkl", pickle_module=dill, encoding='ascii')
    fields = torch.load(path/"fields.pkl", pickle_module=dill, encoding='ascii')
    return Dataset(examples, fields)
