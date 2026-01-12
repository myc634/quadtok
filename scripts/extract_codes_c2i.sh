# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=3 --node_rank=0 \
--master_port=12335 \
extract_codes.py "$@"