#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:12 
Date: January 26, 2020	
"""
import math
import os
import sys
from argparse import ArgumentParser
from math import inf
import time

from torch.autograd import Variable

from criterion.label_smoothed_cross_entropy import LabelSmoothedCrossEntropy
from dataset.data_loader_translation import TranslationDataSet, BySequenceLengthSampler
from dataset.iwslt_data import rebatch_data, subsequent_mask, LabelSmoothing, NoamOpt, SimpleLossCompute
from models.decoding import batch_decode, greedy_decode, generate_beam, beam_decode
from models.transformer import TransformerEncoderDecoder
import torch.nn.functional as F
import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.vocab import WordVocab
from models.utils.model_utils import save_state, load_model_state, get_masks, my_collate, get_perplexity
from optim.lr_warm_up import GradualWarmupScheduler
from options import get_parser


def train(arg):
    model_dir = arg.model
    try:
        os.makedirs(model_dir)
    except OSError:
        pass
    input_file = arg.path
    for lang in (arg.source, arg.target):
        with open(input_file + '.' + lang) as f:
            vocab = WordVocab(f)
            vocab.save_vocab("{}/{}.pkl".format(model_dir, lang))

    vocab_src = WordVocab.load_vocab("{}/{}.pkl".format(model_dir, arg.source))
    vocab_tgt = WordVocab.load_vocab("{}/{}.pkl".format(model_dir, arg.target))

    lr_warmup = arg.lr_warmup
    batch_size = arg.batch_size
    k = arg.dim_model
    h = arg.num_heads
    depth = arg.depth
    max_size=arg.max_length

    data_set = TranslationDataSet(input_file, arg.source, arg.target, vocab_src, vocab_tgt, max_size,
                                  add_sos_and_eos=True)

    # bucket_boundaries = [i * 30 for i in range(20)]
    # sampler = BySequenceLengthSampler(data_set, bucket_boundaries, batch_size)
    # data_loader = DataLoader(data_set, collate_fn=my_collate, batch_sampler=sampler)

    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             collate_fn=my_collate,
                             shuffle=arg.shuffle)

    vocab_size_src = len(vocab_src.stoi)
    vocab_size_tgt = len(vocab_tgt.stoi)

    model = TransformerEncoderDecoder(k, h, dropout=arg.dropout, depth=depth, num_emb=vocab_size_src,
                                      num_emb_target=vocab_size_tgt, max_len=max_size,
                                      mask_future_steps=True)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    start_epoch = load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model,
                                   data_parallel=arg.data_parallel)
    # criterion = LabelSmoothedCrossEntropy(tgt_vocab_size=vocab_size_tgt, label_smoothing=arg.label_smoothing,
    #                                       ignore_index=vocab_tgt.pad_index)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(params=model.parameters(), lr=arg.lr, betas=(0.9, 0.999), eps=1e-8)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arg.num_epochs)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=lr_warmup,
    #                                           after_scheduler=scheduler_cosine)

    criterion = LabelSmoothing(size=len(vocab_tgt.stoi),
                               padding_idx=vocab_tgt.pad_index,
                               smoothing=arg.label_smoothing)

    optimizer = NoamOpt(arg.dim_model, 1, 2000,
                        torch.optim.Adam(model.parameters(),
                                         lr=lr_warmup,
                                         betas=(0.9, 0.98), eps=1e-9))

    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    cuda_condition = torch.cuda.is_available() and not arg.cpu
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    if cuda_condition and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    def truncate_division(x, y):
        return round(x/y, 2)

    previous_best = inf
    for epoch in range(start_epoch, arg.num_epochs):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        # Setting the tqdm progress bar
        # data_iter = tqdm.tqdm(enumerate(data_loader),
        #                       desc="Running epoch: {}".format(epoch),
        #                       total=len(data_loader))
        for i, batch in enumerate(rebatch_data(pad_idx=1, batch=b, device=device) for b in data_loader):
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
    model_dir = arg.model
    vocab_src = WordVocab.load_vocab("{}/{}.pkl".format(model_dir, arg.source))
    vocab_tgt = WordVocab.load_vocab("{}/{}.pkl".format(model_dir, arg.target))
    batch_size = 1
    k = arg.dim_model
    h = arg.num_heads
    depth = arg.depth
    max_size = arg.max_length
    input_file = arg.path
    data_set = TranslationDataSet(input_file, arg.source, arg.target, vocab_src, vocab_tgt, max_size,
                                  add_sos_and_eos=True)

    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             collate_fn=my_collate,
                             shuffle=arg.shuffle)
    vocab_size_src = len(vocab_src.stoi)
    vocab_size_tgt = len(vocab_tgt.stoi)

    model = TransformerEncoderDecoder(k, h, depth=depth, num_emb=vocab_size_src,
                                      num_emb_target=vocab_size_tgt, max_len=max_size,
                                      mask_future_steps=True)

    load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, data_parallel=arg.data_parallel)
    model.eval()

    cuda_condition = torch.cuda.is_available() and not arg.cpu
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    with torch.no_grad():
        for l, batch in enumerate(rebatch_data(pad_idx=1, batch=b, device=device) for b in data_loader):
            # out = greedy_decode(model, batch.src, batch.src_mask, start_symbol=vocab_tgt.sos_index)
            out = beam_decode(model, batch.src, batch.src_mask, batch.src_len,
                               pad_index=vocab_tgt.pad_index,
                               sos_index=vocab_tgt.sos_index,
                               eos_index=vocab_tgt.eos_index)

            # out, lengths = generate_beam(model, batch.src, batch.src_mask, batch.src_len,
            #                              pad_index = vocab_tgt.pad_index,
            #                              sos_index = vocab_tgt.sos_index,
            #                              eos_index = vocab_tgt.eos_index,
            #                              emb_dim=k,
            #                              vocab_size=vocab_size_tgt,
            #                              beam_size=5,
            #                              length_penalty=False,
            #                              early_stopping=False
            #                              )
            print(out.size())
            for i in range(0, out.size(0)):
                print("Source:", end="\t")
                src = list()
                for j in range(0, batch.src.size(1)):
                    sym = vocab_src.itos[batch.src[i, j]]
                    if sym == "<eos>": break
                    src.append(sym)
                print(' '.join(src).replace(' ', '').replace('▁', ' '))
                print("Translation:", end="\t")
                transl = list()
                for j in range(0, out.size(1)):
                    sym = vocab_tgt.itos[out[i, j]]
                    if sym == "<eos>": break
                    transl.append(sym)
                print(' '.join(transl).replace(' ', '').replace('▁', ' '))
                trg = list()
                print("Target:", end="\t")
                for j in range(0, batch.trg.size(1)):
                    sym = vocab_tgt.itos[batch.trg[i, j]]
                    if sym == "<pad>": break
                    trg.append(sym)
                print(' '.join(trg).replace(' ', '').replace('▁', ' '))
                print()
            break


def main():
    options = get_parser()
    if options.train:
        print('Launching training...')
        train(options)
    elif options.decode:
        print('Launching decoding...')
        decode(options)
    else:
        print("Specify either --train or --decode")


if __name__ == "__main__":
    main()
