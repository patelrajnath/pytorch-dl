#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 5:22 PM,  1/24/20
"""

import sys


with open(sys.argv[1]) as file, open(sys.argv[1] + '.paired', 'w') as file_out:
    count = 0
    previous_line = ''
    for line in file:
        previous_line = line
        if count >= 1:
                next_line = line.strip() + '\t' + previous_line.strip()
                file_out.write(next_line + '\n')
                print(next_line)
        count += 1