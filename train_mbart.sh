#!/bin/bash

python train_nmt.py \
	-data mbartdata-bin/demo \
	-save_model mbartdemo-model \
	--enc_layers 1 \
	--batch_type tokens \
	--batch_size 4096 \
	--gpu_ranks 0 \
	--single_pass
