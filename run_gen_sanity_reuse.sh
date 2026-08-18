#!/bin/bash
# 1-GPU, 3-step sanity via reused cp310 env (PYTHONPATH -> base/.venv site-packages).
PY310=/sensei-fs-3/users/yuchengm/.uvpy/cpython-3.10-linux-x86_64-gnu/bin/python3.10
export PYTHONPATH=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/lib/python3.10/site-packages
cd /sensei-fs-3/users/yuchengm/code/quadtok/3level
export CUDA_VISIBLE_DEVICES=0
$PY310 train_gen_varlen.py \
  --config configs/gpt_quadtree_3level.yaml \
  --shards "/tmp/eb_{0..7}.tar" \
  --max_tokens_global 32768 --steps 3 --num_workers 2
echo "SANITY_EXIT $?"
