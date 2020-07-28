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
from options import get_parser


def train(arg):
    model_dir = arg.model
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    train, val, test, SRC, TGT = get_data()

    pad_idx = TGT.vocab.stoi["<blank>"]

    BATCH_SIZE = arg.batch_size
    model_dim = arg.dim_model
    heads = arg.num_heads
    depth = arg.depth
    max_len = arg.max_length

    n_batches = math.ceil(len(train) / BATCH_SIZE)

    train_iter = MyIterator(train, batch_size=BATCH_SIZE,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=arg.dropout,
                                      depth=depth,
                                      num_emb=len(SRC.vocab),
                                      num_emb_target=len(TGT.vocab),
                                      max_len=max_len,
                                      mask_future_steps=True)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    start_epoch = load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model,
                                   data_parallel=arg.data_parallel)

    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=arg.label_smoothing)
    optimizer = NoamOpt(model_dim, 1, 2000, torch.optim.Adam(model.parameters(),
                                                             lr=arg.lr,
                                                             betas=(0.9, 0.98), eps=1e-9))
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

    previous_best = inf

    for epoch in range(start_epoch, arg.num_epochs):
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
    model_dir = arg.model
    train, val, test, SRC, TGT = get_data()
    pad_idx = TGT.vocab.stoi["<blank>"]
    BATCH_SIZE = 1
    model_dim = arg.dim_model
    heads = arg.num_heads
    depth = arg.depth
    max_len = arg.max_length

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

    load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, data_parallel=arg.data_parallel)
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
        for k, batch in enumerate(rebatch(pad_idx, b, device=device) for b in valid_iter):
            # out = greedy_decode(model, batch.src, batch.src_mask, start_symbol=TGT.vocab.stoi["<sos>"])
            start_symbol = TGT.vocab.stoi["<sos>"]

            def beam_search():
                # This is forcing the model to match the source length
                max=batch.ntokens

                beam_size=5
                topk = [[[], .0, None]]  # [sequence, score, key_states]

                memory = model.encoder(batch.src, batch.src_mask)
                input_tokens = torch.ones(1, 1).fill_(start_symbol).type_as(batch.src.data)

                for _ in range(max):
                    candidates = []
                    for i, (seq, score, key_states) in enumerate(topk):
                        # get decoder output
                        if seq:
                            # convert list of tensors to tensor list and add a new dimension for batch
                            input_tokens = torch.stack(seq).unsqueeze(0)

                        # get decoder output
                        out = model.decoder(Variable(input_tokens), memory, batch.src_mask,
                                            Variable(subsequent_mask(input_tokens.size(1)).type_as(batch.src.data)))
                        states = out[:, -1]

                        lprobs, logit = model.generator(states)
                        lprobs[:, pad_idx] = -math.inf  # never select pad
                        # Restrict number of candidates to only twice that of beam size
                        prob, indices = torch.topk(lprobs, 2 * beam_size, dim=1, largest=True, sorted=True)

                        # calculate scores
                        for (idx, val) in zip(indices[0], prob[0]):
                            candidate = [seq + [torch.tensor(idx).to(prob.device)], score + val.item(), i]
                            candidates.append(candidate)

                        # order all candidates by score, select k-best
                        topk = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                return [idx.item() for idx in topk[0][0]]

            out = beam_search()
            print("Source:", end="\t")
            for i in range(1, batch.src.size(1)):
                sym = SRC.vocab.itos[batch.src.data[0, i]]
                if sym == "<eos>": break
                print(sym, end=" ")
            print()
            for i in range(0, 1):
                print("Translation:", end="\t")
                transl = list()
                for j in range(0, len(out)):
                    sym = TGT.vocab.itos[out[j]]
                    if sym == "<eos>": break
                    transl.append(sym)
                print(' '.join(transl))
            print()
            # print("Translation:", end="\t")
            # for i in range(1, out.size(1)):
            #     sym = TGT.vocab.itos[out[0, i]]
            #     if sym == "<eos>": break
            #     print(sym, end=" ")
            # print()
            print("Target:", end="\t")
            for i in range(1, batch.trg.size(1)):
                sym = TGT.vocab.itos[batch.trg.data[0, i]]
                if sym == "<eos>": break
                print(sym, end=" ")
            print()

            if k==10:
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
