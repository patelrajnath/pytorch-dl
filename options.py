#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 23:05 
Date: February 20, 2020	
"""
from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', dest="train", action='store_true', help="Enable for training")
    mode.add_argument('--decode', dest="decode", action='store_true')

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=30, type=int)

    parser.add_argument("--data-parallel",
                        dest='data_parallel',
                        action='store_true',
                        help="Enable it for decoding if training was donw with multi-gpu")

    parser.add_argument("--shuffle",
                        dest='shuffle',
                        action='store_true',
                        help="Enable data shuffling")

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

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

    parser.add_argument("--model", dest="model",
                        help="model directory",
                        default='nmt')

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
    return options
