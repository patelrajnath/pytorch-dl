#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 08:53 
Date: January 22, 2020	
"""
import numpy as np

a = [23, 44, 55, 66, 67, 23, 45, 54, 63]


def sum_digit(digits):
    return sum([int(i) for i in str(digits)])


d = {}
for item in a:
    num_sum = sum_digit(item)
    if num_sum in d:
        d[num_sum].append(item)
    else:
        d[num_sum] = [item]

# print(d)
# print(sorted(d, reverse=True))
# print(d[13])

a = np.array([[[10, 11, 12], [13, 14, 15], [16, 17, 18]],
               [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
               [[30, 31, 32], [33, 34, 35], [36, 37, 38]]])

print(a[0, 1])