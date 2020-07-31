#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 6:59 AM,  7/30/20
"""
import os
import time
from math import inf

from torch import nn

from dataset.iwslt_data import rebatch, rebatch_onmt, SimpleLossCompute, NoamOpt, LabelSmoothing
from models.transformer import TransformerEncoderDecoder
from models.utils.model_utils import load_model_state, save_state, get_perplexity

"""Train models."""
import torch

import onmt.opts as opts

from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, patch_fields, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple


def train(opts):
    ArgumentParser.validate_train_opts(opts)
    ArgumentParser.update_model_opts(opts)
    ArgumentParser.validate_model_opts(opts)

    set_random_seed(opts.seed, False)

    # Load checkpoint if we resume from a previous training.
    if opts.train_from:
        logger.info('Loading checkpoint from %s' % opts.train_from)
        checkpoint = torch.load(opts.train_from,
                                map_location=lambda storage, loc: storage)
        logger.info('Loading vocab from checkpoint at %s.' % opts.train_from)
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(opts.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opts.model_type, dynamic_dict=opts.copy_attn)
    else:
        fields = vocab

    src_vocab_size = len(fields['src'].base_field.vocab)
    trg_vocab_size = len(fields['tgt'].base_field.vocab)

    # patch for fields that may be missing in old data/model
    patch_fields(opts, fields)

    if len(opts.data_ids) > 1:
        train_shards = []
        for train_id in opts.data_ids:
            shard_base = "train_" + train_id
            train_shards.append(shard_base)
        train_iter = build_dataset_iter_multiple(train_shards, fields, opts)
    else:
        if opts.data_ids[0] is not None:
            shard_base = "train_" + opts.data_ids[0]
        else:
            shard_base = "train"
        train_iter = build_dataset_iter(shard_base, fields, opts)

    pad_idx = 1

    model_dir = opts.save_model
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    model_dim = opts.state_dim
    heads = opts.heads
    depth = opts.enc_layers
    max_len = 100

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=opts.dropout[0],
                                      depth=depth,
                                      num_emb=src_vocab_size,
                                      num_emb_target=trg_vocab_size,
                                      max_len=max_len,
                                      mask_future_steps=True)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    start_steps = load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model,
                                   data_parallel=False)

    criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=pad_idx, smoothing=opts.label_smoothing)
    optimizer = NoamOpt(model_dim, 1, 2000, torch.optim.Adam(model.parameters(),
                                                             lr=opts.learning_rate,
                                                             betas=(0.9, 0.98), eps=1e-9))
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    cuda_condition = torch.cuda.is_available() and opts.gpu_ranks
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    if cuda_condition and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    previous_best = inf
    # start steps defines if training was intrupted
    global_steps = start_steps
    iterations = 0
    while global_steps <= opts.train_steps:
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        iterations += 1
        for i, batch in enumerate(rebatch_onmt(pad_idx, b, device=device) for b in train_iter):
            global_steps += 1
            model.train()
            out = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)
            loss = compute_loss(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % opts.report_every == 0 and i > 0:
                elapsed = time.time() - start
                print("Epoch %d Step: %d Loss: %f PPL: %f Tokens per Sec: %f" %
                      (iterations, i, loss / batch.ntokens, get_perplexity(loss / batch.ntokens), tokens / elapsed))
                start = time.time()
                tokens = 0
                # checkpoint = "checkpoint.{}.".format(total_loss / total_tokens) + 'epoch' + str(epoch) + ".pt"
                # save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, epoch)
        loss_average = total_loss / total_tokens
        checkpoint = "checkpoint.{}.".format(loss_average) + 'epoch' + str(iterations) + ".pt"
        save_state(os.path.join(model_dir, checkpoint), model, criterion, optimizer, global_steps, fields, opts)

        if previous_best > loss_average:
            save_state(os.path.join(model_dir, 'checkpoints_best.pt'), model, criterion, optimizer, global_steps, fields, opts)
            previous_best = loss_average


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()
