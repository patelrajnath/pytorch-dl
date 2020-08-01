#!/bin/bash
mkdir mbartdata-bin -p
python preprocess.py \
	-train_src sample-data/mbart/train-mbart.txt \
	-train_tgt sample-data/mbart/train-mbart.txt \
	-valid_src sample-data/mbart/valid-mbart.txt \
	-valid_tgt sample-data/mbart/valid-mbart.txt \
	-save_data mbartdata-bin/demo \
	--mbart_masking
