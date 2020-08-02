#!/bin/bash

python train_onmt.py \
	-data data-bin/demo \
	-save_model mbartdemo-model \
	--batch_type tokens \
	--enc_layers 1 \
	--batch_size 4096 \
	--gpu_ranks 0 \
	--single_pass