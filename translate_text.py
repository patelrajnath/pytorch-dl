#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 7:48 AM,  7/31/20
"""
import os
import time

import torch

from dataset.iwslt_data import rebatch_onmt, rebatch_source_only
from models.decoding import batched_beam_search, greedy_decode
from models.transformer import TransformerEncoderDecoder
from models.utils.model_utils import load_model_state
from onmt import opts, inputters
from onmt.inputters import old_style_vocab, load_old_vocab
from onmt.inputters.inputter import patch_fields
from onmt.utils import set_random_seed
from onmt.utils.parse import ArgumentParser


def translate_file(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    with open(opt.src) as input:
        src = input.readlines()

    src_reader = inputters.str2reader['text'].from_opt(opt)
    src_data = {"reader": src_reader, "data": src, "dir": ''}

    _readers, _data, _dir = inputters.Dataset.config(
        [('src', src_data)])

    vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # patch for fields that may be missing in old data/model
    patch_fields(opt, fields)

    # corpus_id field is useless here
    if fields.get("corpus_id", None) is not None:
        fields.pop('corpus_id')

    data = inputters.Dataset(fields, readers=_readers, dirs=_dir, data=_data, sort_key=inputters.str2sortkey['text'])

    data_iter = inputters.OrderedIterator(
        dataset=data,
        batch_size=1,
        train=False,
        sort=False,
        sort_within_batch=True,
        shuffle=False
    )

    src_vocab = fields['src'].base_field.vocab
    trg_vocab = fields['tgt'].base_field.vocab

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
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

    start_steps = load_model_state(os.path.join(model_dir, 'checkpoints_best.pt'), model,
                                   data_parallel=False)
    model.eval()

    cuda_condition = torch.cuda.is_available() and opt.gpu_ranks
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    if cuda_condition:
        model.cuda()

    with torch.no_grad():
        translated = list()
        reference = list()
        start = time.time()
        for k, batch in enumerate(rebatch_source_only(pad_idx, b, device=device) for b in data_iter):
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
            text_transl = " ".join(transl).replace("@@ ", '')
            translated.append(text_transl)

            print(text_transl)

            # print()
            # print("Target:", end="\t")
            # ref = list()
            # for i in range(1, batch.trg.size(1)):
            #     sym = trg_vocab.itos[batch.trg.data[0, i]]
            #     if sym == "</s>": break
            #     ref.append(sym)
            # reference.append(" ".join(ref))

            # if k == 1:
            #     break

    with open('test-beam-decode.de-en.en', 'w', encoding='utf8') as outfile:
        outfile.write('\n'.join(translated))
    # with open('valid-ref.de-en.en', 'w', encoding='utf-8') as outfile:
    #     outfile.write('\n'.join(reference))
    print('Time elapsed:{}'.format(time.time() - start))


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    translate_file(opt)


if __name__ == "__main__":
    main()