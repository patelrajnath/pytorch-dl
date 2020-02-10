#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

import numpy
import torch
from torch.nn import functional

from models.utils.model_utils import get_masks

torch.manual_seed(200)

x = torch.rand(2, 3)
y = torch.Tensor(2, 3)

z = x + x
# print(z)
torch.add(x, x, out=y)
# print(y)
# print(torch.is_tensor(numpy.random.rand(2, 3)))
# print(x.matmul(y.transpose(0, 1)))

x = torch.Tensor([1, 2, 3])
y = (x != 1).unsqueeze(-2)
r = torch.unsqueeze(x, 0)       # Size: 1x3
# print(x.size(), r.size())
r = torch.unsqueeze(x, 1)
# print(x.size(), r.size())

x = torch.Tensor([[1, 2], [3, 4]])
# print(functional.softmax(x, dim=0))
attn_weights = torch.rand(4, 4, 4, 4)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_mask(tensor):
    _mask = None
    dim = tensor.size(-1)
    if _mask is None:
        _mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
    if _mask.size(0) < dim:
        _mask = torch.triu(fill_with_neg_inf(_mask.resize_(dim, dim)), 1)
    return _mask[:dim, :dim]


attn_weights += buffered_mask(attn_weights).unsqueeze(0)
# print(attn_weights)

attn_weights = torch.rand(4, 4, 4)
b, h, w = attn_weights.size()
# print(attn_weights)

mask_diagonal = False
maskval=float('-inf')
# indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
# x = attn_weights[:, indices[0], indices[1]] = maskval
# print(attn_weights)

x = torch.Tensor([[1, 2, 4, 0], [3, 4, 0, 0], [1, 2, 23, 45]])
lenghts = torch.Tensor([3, 2, 4])
slen, bs = x.size()
print(get_masks(slen, y, causal=True))
