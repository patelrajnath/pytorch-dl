#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:12 
Date: January 26, 2020	
"""

import os
from argparse import ArgumentParser

from torch.autograd import Variable

from dataset.data_loader_translation import TranslationDataSet
from models.transformer import TransformerEncoderDecoder

import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.vocab import WordVocab
from models.utils.model_utils import save_state, load_model_state


def go(arg):
    input_file = arg.path
    for lang in (arg.source, arg.target):
        with open(input_file + '.' + lang) as f:
            vocab = WordVocab(f)
            vocab.save_vocab("sample-data/{}.pkl".format(lang))

    vocab_src = WordVocab.load_vocab("sample-data/{}.pkl".format(arg.source))
    vocab_tgt = WordVocab.load_vocab("sample-data/{}.pkl".format(arg.target))

    lr_warmup = arg.lr_warmup
    batch_size = arg.batch_size
    k = arg.embedding_size
    h = arg.num_heads
    depth = arg.depth
    max_size=arg.max_length
    modeldir = "nmt"
    data_set = TranslationDataSet(input_file, arg.source, arg.target, vocab_src, vocab_tgt, max_size)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    vocab_size_src = len(vocab_src.stoi)
    vocab_size_tgt = len(vocab_tgt.stoi)
    model = TransformerEncoderDecoder(k, h, depth=depth, num_emb=vocab_size_src,
                                      num_emb_target=vocab_size_tgt, max_len=max_size)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = Adam(lr=arg.lr, params=model.parameters())
    lr_schedular = lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    if cuda_condition and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    for epoch in range(arg.num_epochs):
        avg_loss = 0
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="Running epoch: {}".format(epoch),
                              total=len(data_loader))
        for i, data in data_iter:
            data = {key: value.to(device) for key, value in data.items()}
            src_tokens, tgt_tokens = data
            decoder_out = model(data[src_tokens], data[tgt_tokens])
            loss = criterion(decoder_out.transpose(1, 2), data[tgt_tokens])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedular.step(epoch)
            avg_loss += loss.item()
            if i % arg.wait == 0 and i > 0:
                try:
                    os.makedirs(modeldir)
                except OSError:
                    pass
                checkpoint = "checkpoint.{}.".format(avg_loss/i) + str(epoch) + ".pt"
                save_state(os.path.join(modeldir, checkpoint), model, criterion, optimizer, epoch)
        try:
            os.makedirs(modeldir)
        except OSError:
            pass
        checkpoint = "checkpoint.{}.".format(avg_loss / len(data_iter)) + str(epoch) + ".pt"
        save_state(os.path.join(modeldir, checkpoint), model, criterion, optimizer, epoch)
        print('Average loss: {}'.format(avg_loss / len(data_iter)))


def decode(arg):
    vocab_src = WordVocab.load_vocab("sample-data/{}.pkl".format(arg.source))
    vocab_tgt = WordVocab.load_vocab("sample-data/{}.pkl".format(arg.target))
    print(len(vocab_tgt.counter))
    print(len(vocab_src.counter))
    print(len(vocab_src.itos))
    print(len(vocab_tgt.itos))

    batch_size = 1
    k = arg.embedding_size
    h = arg.num_heads
    depth = arg.depth
    max_size = arg.max_length
    modeldir = "/home/raj/PycharmProjects/models/nmt"
    input_file = arg.path
    data_set = TranslationDataSet(input_file, arg.source, arg.target, vocab_src, vocab_tgt, max_size)

    data_loader = DataLoader(data_set, batch_size=batch_size)
    vocab_size_src = len(vocab_src.stoi)
    vocab_size_tgt = len(vocab_tgt.stoi)

    model = TransformerEncoderDecoder(k, h, depth=depth, num_emb=vocab_size_src,
                                      num_emb_target=vocab_size_tgt, max_len=max_size)

    load_model_state(os.path.join(modeldir, 'checkpoint_best.pt'), model)

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    # Setting the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc="Decoding",
                          total=len(data_loader))

    def greedy_decode(model, src, max_len, start_symbol):
        memory = model.encoder(src)
        ys = [start_symbol]
        padding = [vocab_tgt.pad_index for _ in range(max_len - len(ys))]
        ys.extend(padding)
        ys = torch.tensor(ys).unsqueeze(0)
        for i in range(max_len - 1):
            out = model.decoder(ys, memory)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys[0, i] = next_word
        return ys

    with torch.no_grad():
        for i, data in data_iter:
            data = {key: value.to(device) for key, value in data.items()}
            src_tokens, tgt_tokens = data
            out = greedy_decode(model, data[src_tokens], max_len=80, start_symbol=vocab_tgt.sos_index)
            print(out)
            print("Translation:", end="\t")
            for i in range(1, out.size(1)):
                sym = vocab_tgt.itos[out[0, i]]
                if sym == "<pad>": break
                print(sym, end=" ")
            print()
            print("Target:", end="\t")
            for i in range(1, data[tgt_tokens].size(1)):
                sym = vocab_tgt.itos[data[tgt_tokens][0, i]]
                if sym == "<pad>": break
                print(sym, end=" ")
            print()
            break


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-P", "--path", dest="path",
                        help="sample training file",
                        default='sample-data/europarl.enc')
    parser.add_argument("-S", "--src", dest="source",
                        help="source language",
                        default='en')
    parser.add_argument("-T", "--tgt", dest="target",
                        help="target language",
                        default='it')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=512, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=80, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=1000, type=int)

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

    # go(options)
    decode(options)

