#!/bin/bash

python pretrain_roberta.py \
	-data roberta-data-bin/demo \
	-save_model roberta-demo-model \
	--batch_type tokens \
	--enc_layers 1 \
	--batch_size 1024 \
	--gpu_ranks 0 \
	--single_pass
