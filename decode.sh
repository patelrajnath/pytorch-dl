#!/bin/bash

python decode_nmt.py -data data-bin/demo \
	-save_model demo-model \
	--valid_batch_size 1 \
  --gpu_ranks 0
