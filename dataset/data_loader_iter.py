#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 09:11 
Date: February 20, 2020	
"""
from torch.utils.data.dataset import IterableDataset


class MyIterableDataset(IterableDataset):

    def __init__(self, filename):
        # Store the filename in object's memory
        self.filename = filename
        # And that's it, we no longer need to store the contents in the memory

    def preprocess(self, text):
        # Do something with text here
        text_pp = text.lower().strip()
        return text_pp

    def line_mapper(self, line):
        # Splits the line into text and label and applies preprocessing to the text
        text, label = line.split(',')
        text = self.preprocess(text)

        return text, label

    def __iter__(self):
        # Create an iterator
        file_itr = open(self.filename)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr
