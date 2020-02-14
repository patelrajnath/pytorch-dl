#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 12:22 PM,  2/13/20
"""

import sys

map = {
    'ä': 'ae',
    'ö': 'oe',
    'ü': 'ue',
    'Ä': 'Ae',
    'Ö': 'Oe',
    'Ü': 'Ue',
    'ß': 'ss'
}

all_caps_map = {
    'Ä': 'AE',
    'Ö': 'OE',
    'Ü': 'UE',
    'ß': 'SS'
}

with open(sys.argv[1]) as fin, open(sys.argv[1]+'.ascii.txt', 'w') as fout, open(sys.argv[1]+'.log', 'w') as logfile:
    for line in fin:
        ascii_flag = False
        original = line
        tokens = line.split()
        tokens_updated = list()
        for i in range(0, len(tokens), 2):
            word = tokens[i]
            if word.isupper():
                for latin in all_caps_map:
                    if latin in word:
                        word = word.replace(latin, all_caps_map[latin])
                        ascii_flag = True
            else:
                for latin in map:
                    if latin in word:
                        word = word.replace(latin, map[latin])
                        ascii_flag = True
            tokens[i] = word

        if ascii_flag:
            logfile.write(original)
            fout.write(" ".join(tokens) + '\n')
