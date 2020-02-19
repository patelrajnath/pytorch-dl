#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""
import os
from argparse import ArgumentParser

from dataset.iwslt_data import LabelSmoothing, NoamOpt, SimpleLossCompute, rebatch_mbert
from models.transformer import TransformerEncoderDecoder

import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.data_loader_mbert import MBertDataSet
from dataset.vocab import WordVocab
from models.utils.model_utils import save_state, my_collate


def go(arg):
    input_file = arg.path
    with open(input_file) as f:
        vocab = WordVocab(f)
        vocab.save_vocab("sample-data/vocab.pkl")

    vocab = WordVocab.load_vocab("sample-data/vocab.pkl")

    lr_warmup = arg.lr_warmup
    batch_size = arg.batch_size
    k = arg.dim_model
    h = arg.num_heads
    depth = arg.depth
    max_size=arg.max_length
    modeldir = "bert"
    data_set = MBertDataSet(input_file, vocab, max_size)

    data_loader = DataLoader(data_set, batch_size=batch_size,
                             collate_fn=my_collate,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=False,
                             shuffle=True
                             )

    vocab_size = len(vocab.stoi)
    model = TransformerEncoderDecoder(k, h, depth=depth, num_emb=vocab_size, num_emb_target=vocab_size, max_len=max_size)

    # criterion = nn.NLLLoss(ignore_index=0)
    # optimizer = Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    #                  weight_decay=0, amsgrad=False)
    # lr_schedular = lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    criterion = LabelSmoothing(size=len(vocab.stoi), padding_idx=1, smoothing=0.1)
    optimizer = NoamOpt(arg.dim_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    use_cuda = torch.cuda.is_available() and not arg.cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model.to(device)

    if use_cuda and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    for epoch in range(arg.num_epochs):
        avg_loss = 0
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="Running epoch: {}".format(epoch),
                              total=len(data_loader))
        for i, batch in enumerate(rebatch_mbert(pad_idx=vocab.pad_index, batch=b, device=device) for b in data_loader):

            # For mBert we use both source and target the same but the target-mask as its to skip the future steps
            out = model(batch.src, batch.src_mask, batch.src, batch.trg_mask)
            loss = compute_loss(out, batch.trg, batch.ntokens)
            avg_loss += loss.item()
            if i % arg.wait == 0 and i > 0:
                checkpoint = "checkpoint.{}.".format(avg_loss/i) + str(epoch) + ".pt"
                try:
                    os.makedirs(modeldir)
                except OSError:
                    pass
                save_state(os.path.join(modeldir, checkpoint), model, criterion, optimizer, epoch)
        print('Average loss: {}'.format(avg_loss / len(data_iter)))


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
                        default='sample-data/europarl.enc.en')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--cpu", dest="cpu",
                        help="Use CPU computation.).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--dim_model", dest="dim_model",
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

    go(options)

