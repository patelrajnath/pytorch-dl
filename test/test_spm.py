#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 4:40 PM,  2/20/20
"""
import sentencepiece as spm


def concat_src_trg(prefix, src, trg):
    # Reading data from file1
    with open(prefix + '.' + src) as fp:
        data = fp.read()

        # Reading data from file2
    with open(prefix + '.' + trg) as fp:
        data2 = fp.read()

        # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += data2
    output = prefix + '.' + src + '-' + trg
    with open(output, 'w') as fp:
        fp.write(data)
    return output


input_file = '.data/iwslt//de-en/train.de-en'
concatenated = concat_src_trg(input_file, 'de', 'en')

input_sentnce_size = 100000
spm.SentencePieceTrainer.Train('--input={} --input_sentence_size={} --shuffle_input_sentence=true '
                               '--model_prefix=m --vocab_size={}'.format(concatenated, input_sentnce_size, 32000))

