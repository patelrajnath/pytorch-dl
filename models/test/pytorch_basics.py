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
r = torch.unsqueeze(x, 0)       # Size: 1x3
# print(x.size(), r.size())
r = torch.unsqueeze(x, 1)
# print(x.size(), r.size())

x = torch.Tensor([[1, 2], [3, 4]])
print(functional.softmax(x, dim=0))