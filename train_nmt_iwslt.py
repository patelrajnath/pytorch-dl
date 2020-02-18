#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 15:47 
Date: February 15, 2020	
"""
import math
import os
from argparse import ArgumentParser
from math import inf
import time

import torch
import tqdm
from torch import nn
from torch.autograd import Variable

from dataset.iwslt_data import get_data, MyIterator, batch_size_fn, rebatch, SimpleLossCompute, LabelSmoothing, \
    subsequent_mask

from dataset.iwslt_data import NoamOpt
from models.decoding import greedy_decode
from models.transformer import TransformerEncoderDecoder
from models.utils.model_utils import save_state, load_model_state, get_perplexity
from optim.lr_warm_up import GradualWarmupScheduler


def train(arg):
    train, val, test, SRC, TGT = get_data()

    pad_idx = TGT.vocab.stoi["<blank>"]

    BATCH_SIZE = arg.batch_size
    model_dim = arg.dim_model
    heads = arg.num_heads
    depth = arg.depth
    max_len = arg.max_length

    n_batches = math.ceil(len(train) / BATCH_SIZE)

    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=arg.dropout, depth=depth, num_emb=len(SRC.vocab),
                                      num_emb_target=len(TGT.vocab), max_len=max_len,
                                      mask_future_steps=True)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    optimizer = NoamOpt(model_dim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=1000,
    #                                           after_scheduler=scheduler_cosine)

    cuda_condition = torch.cuda.is_available() and not arg.cpu
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    if cuda_condition and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    model_dir = "transformer-model"
    try:
        os.makedirs(model_dir)
    except OSError:
        pass
    previous_best = inf

    for epoch in range(1, arg.num_epochs):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(rebatch(pad_idx, b, device=device) for b in train_iter):
            model.train()
            # bs = batch.batch_size
            # tgt_lengths = (trg != pad_idx).data.sum(dim=1)
            # src_lengths = (src != pad_idx).data.sum(dim=1)
            # batch_ntokens = tgt_lengths.sum()
            # src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
            out = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)
            loss = compute_loss(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % arg.wait == 0 and i > 0:
                elapsed = time.time() - start
                print("Epoch %d Step: %d Loss: %f PPL: %f Tokens per Sec: %f" %
                      (epoch, i, loss / batch.ntokens, get_perplexity(loss / batch.ntokens), tokens / elapsed))
                start = time.time()
                tokens = 0
                # checkpoint = "checkpoint.{}.".format(total_loss / total_tokens) + 'epoch' + str(epoch) + ".pt"
                # save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, epoch)
        loss_average = total_loss / total_tokens
        checkpoint = "checkpoint.{}.".format(loss_average) + 'epoch' + str(epoch) + ".pt"
        save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, epoch)

        if previous_best > loss_average:
            save_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, criterion, optimizer, epoch)
            previous_best = loss_average


def decode(arg):
    train, val, test, SRC, TGT = get_data()
    pad_idx = TGT.vocab.stoi["<blank>"]
    BATCH_SIZE = arg.batch_size
    model_dim = arg.dim_model
    heads = arg.num_heads
    depth = arg.depth
    max_len = arg.max_length
    model_dir = "transformer-model"

    n_batches = math.ceil(len(train) / BATCH_SIZE)

    train_iter = MyIterator(train, batch_size=BATCH_SIZE,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=arg.dropout, depth=depth,
                                      num_emb=len(SRC.vocab),
                                      num_emb_target=len(TGT.vocab), max_len=max_len,
                                      mask_future_steps=True)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, data_parallel=True)
    model.eval()

    cuda_condition = torch.cuda.is_available() and not arg.cpu
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    # Setting the tqdm progress bar
    # data_iter = tqdm.tqdm(enumerate(data_loader),
    #                       desc="Decoding",
    #                       total=len(data_loader))

    with torch.no_grad():
        for k, batch in enumerate(valid_iter):
            src = batch.src.transpose(0, 1)[:1]
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, start_symbol=TGT.vocab.stoi["<sos>"])
            print("Translation:", end="\t")
            for i in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[0, i]]
                if sym == "<eos>": break
                print(sym, end=" ")
            print()
            print("Target:", end="\t")
            for i in range(1, batch.trg.size(0)):
                sym = TGT.vocab.itos[batch.trg.data[i, 0]]
                if sym == "<eos>": break
                print(sym, end=" ")
            print()
            break


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=30, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4000, type=int)

    parser.add_argument("--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0005, type=float)

    parser.add_argument("--dropout",
                        dest="dropout",
                        help="Learning rate",
                        default=0.1, type=float)
    parser.add_argument("--label-smoothing",
                        dest="label_smoothing",
                        help="Label smoothing rate",
                        default=0.1, type=float)

    parser.add_argument("-P", "--path", dest="path",
                        help="sample training file",
                        default='sample-data/europarl.enc')
    parser.add_argument("-S", "--src", dest="source",
                        help="source language",
                        default='it')
    parser.add_argument("-T", "--tgt", dest="target",
                        help="target language",
                        default='en')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("--cpu", dest="cpu",
                        help="Use cpu for training.",
                        action="store_true")

    parser.add_argument("-D", "--dim-model", dest="dim_model",
                        help="model size.",
                        default=512, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=160, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=4000, type=int)

    parser.add_argument("--wait",
                        dest="wait",
                        help="Learning rate warmup.",
                        default=1000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    train(options)
    # decode(options)
