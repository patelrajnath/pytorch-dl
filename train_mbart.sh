#!/bin/bash

python train_onmt.py \
	-data mbartdata-bin/demo \
	-save_model mbartdemo-model \
	--batch_type tokens \
	--batch_size 1096 \
	--gpu_ranks 0 \
	--single_pass