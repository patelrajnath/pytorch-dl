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
from models.decoding import beam_search, batched_beam_search
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


def decode(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    src_vocab = fields['src'].base_field.vocab
    trg_vocab = fields['tgt'].base_field.vocab

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    # patch for fields that may be missing in old data/model
    patch_fields(opt, fields)

    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    pad_idx = 1

    model_dir = opt.save_model
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    model_dim = opt.state_dim
    heads = opt.heads
    depth = opt.enc_layers
    max_len = 100

    model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=opt.dropout[0],
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

    criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=pad_idx, smoothing=opt.label_smoothing)
    optimizer = NoamOpt(model_dim, 1, 2000, torch.optim.Adam(model.parameters(),
                                                             lr=opt.learning_rate,
                                                             betas=(0.9, 0.98), eps=1e-9))
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    cuda_condition = torch.cuda.is_available() and opt.gpu_ranks
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    if cuda_condition and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    with torch.no_grad():
        translated = list()
        reference = list()
        for k, batch in enumerate(rebatch_onmt(pad_idx, b, device=device) for b in valid_iter):
            print('Processing: {0}'.format(k))
            start_symbol = trg_vocab.stoi["<sos>"]
            # out = greedy_decode(model, batch.src, batch.src_mask, start_symbol=start_symbol)
            out = beam_search(model, batch.src, batch.src_mask, start_symbol=start_symbol, pad_symbol=pad_idx,
                              max=batch.ntokens + 10)
            # print("Source:", end="\t")
            # for i in range(1, batch.src.size(1)):
            #     sym = SRC.vocab.itos[batch.src.data[0, i]]
            #     if sym == "<eos>": break
            #     print(sym, end=" ")
            # print()
            # print("Translation:", end="\t")

            transl = list()
            start_idx = 0  # for greedy decoding the start index should be 1 that will exclude the <sos> symbol
            for i in range(start_idx, out.size(1)):
                sym = trg_vocab.itos[out[0, i]]
                if sym == "</s>": break
                transl.append(sym)
            translated.append(' '.join(transl))

            # print()
            # print("Target:", end="\t")
            ref = list()
            for i in range(1, batch.trg.size(1)):
                sym =  trg_vocab.itos[batch.trg.data[0, i]]
                if sym == "</s>": break
                ref.append(sym)
            reference.append(" ".join(ref))

            if k == 1:
                break

        with open('valid-beam-decode-test.de-en.en', 'w') as outfile:
            outfile.write('\n'.join(translated))
        with open('valid-ref.de-en.en', 'w') as outfile:
            outfile.write('\n'.join(reference))


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    decode(opt)


if __name__ == "__main__":
    main()
