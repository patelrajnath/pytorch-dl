#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

import torch
import numpy as np

from models.transformer import SelfAttention, TransformerBlock

k=32
h=8
w=20
b=16

x = np.random.rand(b, w, k)

model = TransformerBlock(k, h)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x).float().to(device)
model(x_train_tensor)