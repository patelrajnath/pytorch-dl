#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""
import logging
import math
import os
import traceback
from copy import deepcopy

import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.serialization import default_restore_location

from models.transformer import TransformerEncoderDecoder
from onmt import inputters
from onmt.inputters.inputter import patch_fields
from onmt.utils.parse import ArgumentParser


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """
    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


def get_masks(slen, lengths, causal=False):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, model, criterion, optimizer, num_updates,
               fields=None, opts=None, optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}

    if fields:
        vocab = deepcopy(fields)
        for side in ["src", "tgt"]:
            keys_to_pop = []
            if hasattr(vocab[side], "fields"):
                unk_token = vocab[side].fields[0][1].vocab.itos[0]
                for key, value in vocab[side].fields[0][1].vocab.stoi.items():
                    if value == 0 and key != unk_token:
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    vocab[side].fields[0][1].vocab.stoi.pop(key, None)
    else:
        vocab = {}

    if opts:
        opts = opts
    else:
        opts = {}

    print("Saving checkpoint at-", filename)
    state_dict = {
        'model': model.state_dict(),
        'num_updates': num_updates,
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
            }
        ],
        'extra_state': extra_state,
        'vocab': vocab,
        'model_opts': opts,
    }
    torch_persistent_save(state_dict, filename)


def load_model_state(filename, opts, model=None, data_parallel=True):
    if not os.path.exists(filename):
        print("Starting training from scratch.")
        return 0

    print("Loading model from checkpoints", filename)
    checkpoint = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))

    if model:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # create new OrderedDict that does not contain `module.`
        if data_parallel:
            for k, v in checkpoint['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['model']

        # load model parameters
        try:
            model.load_state_dict(new_state_dict)
        except Exception:
            raise Exception('Cannot load model parameters from checkpoint, '
                            'please ensure that the architectures match')
        return checkpoint['num_updates']

    else:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint['model_opts'])
        vocab = checkpoint['vocab']
        if inputters.old_style_vocab(vocab):
            fields = inputters.load_old_vocab(
                vocab, opts.data_type, dynamic_dict=model_opt.copy_attn
            )
        else:
            fields = vocab

        # patch for fields that may be missing in old data/model
        patch_fields(model_opt, fields)

        src_vocab = fields['src'].base_field.vocab
        trg_vocab = fields['tgt'].base_field.vocab

        src_vocab_size = len(src_vocab)
        trg_vocab_size = len(trg_vocab)

        model_dim = model_opt.state_dim
        heads = model_opt.heads
        depth = model_opt.enc_layers
        max_len = 100

        model = TransformerEncoderDecoder(k=model_dim, heads=heads, dropout=model_opt.dropout[0],
                                          depth=depth,
                                          num_emb=src_vocab_size,
                                          num_emb_target=trg_vocab_size,
                                          max_len=max_len,
                                          mask_future_steps=True)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # create new OrderedDict that does not contain `module.`
        if data_parallel:
            for k, v in checkpoint['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['model']

        # load model parameters
        try:
            model.load_state_dict(new_state_dict)
        except Exception:
            raise Exception('Cannot load model parameters from checkpoint, '
                            'please ensure that the architectures match')
        return checkpoint['num_updates'], model, fields


def my_collate(batch):
    # batch contains a list of dict of structure [{"source": source, "target": target}]

    lengths_source = torch.tensor([t["source"].shape[0] for t in batch])
    lengths_target = torch.tensor([t["target"].shape[0] for t in batch])

    source = [item["source"] for item in batch]
    source = pad_sequence(source, padding_value=1).transpose(0, 1)

    targets = [item["target"] for item in batch]
    targets = pad_sequence(targets, padding_value=1).transpose(0, 1)

    return source, targets, lengths_source, lengths_target


def get_perplexity(loss):
    try:
        return math.pow(2, loss)
    except OverflowError:
        return float('inf')
