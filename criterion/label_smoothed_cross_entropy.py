#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:06 
Date: February 02, 2020	
"""

#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
 Copyright (c) 2018 Raj Nath Patel
 Licensed under the GNU Public License.
 Author: Raj Nath Patel
 Email: patelrajnath (at) gmail (dot) com
 Created: 25/May/2018 03:55
 """
import torch


from torch import nn
from torch.autograd import Variable


class LabelSmoothedCrossEntropy(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
