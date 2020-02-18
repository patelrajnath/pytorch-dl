#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 12:21 PM,  1/21/20
"""
import numpy as np


def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


print(softmax([0.01, 0.001, 0.002, 0.003, 0.01, 0.002]))