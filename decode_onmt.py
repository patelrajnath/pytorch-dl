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
from models.decoding import beam_search, batched_beam_search, greedy_decode
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

    pad_idx = 1

    model_dir = opt.save_model
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    start_steps, model, fields = load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), opts,
                                                  data_parallel=False)
    model.eval()
    
    src_vocab = fields['src'].base_field.vocab
    trg_vocab = fields['tgt'].base_field.vocab

    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    cuda_condition = torch.cuda.is_available() and opt.gpu_ranks
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    with torch.no_grad():
        translated = list()
        reference = list()
        start = time.time()
        for k, batch in enumerate(rebatch_onmt(pad_idx, b, device=device) for b in valid_iter):
            print('Processing: {0}'.format(k))
            start_symbol = trg_vocab.stoi["<s>"]

            # out = greedy_decode(model, batch.src, batch.src_mask, start_symbol=start_symbol)
            # out = beam_search(model, batch.src, batch.src_mask,
            #                           start_symbol=start_symbol, pad_symbol=pad_idx,
            #                           max=batch.ntokens + 10)
            out = batched_beam_search(model, batch.src, batch.src_mask,
                                      start_symbol=start_symbol, pad_symbol=pad_idx,
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

#             if k == 1:
#                 break

        with open('valid-beam-decode-test.de-en.en', 'w', encoding='utf8') as outfile:
            outfile.write('\n'.join(translated))
        with open('valid-ref.de-en.en', 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(reference))
        print('Time elapsed:{}'.format(time.time()- start))


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
