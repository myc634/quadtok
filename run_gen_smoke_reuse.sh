#!/bin/bash
# 8-GPU, 100-step smoke @ max_tokens_global = 262144*4 = 1,048,576 (131,072/GPU),
# via reused cp310 env (PYTHONPATH -> base/.venv site-packages; base venv's torchrun
# console-script has a broken shebang on this pod, so use `python -m torch.distributed.run`).
PY310=/sensei-fs-3/users/yuchengm/.uvpy/cpython-3.10-linux-x86_64-gnu/bin/python3.10
export PYTHONPATH=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/lib/python3.10/site-packages
cd /sensei-fs-3/users/yuchengm/code/quadtok/3level
export OMP_NUM_THREADS=8
$PY310 -m torch.distributed.run --nproc_per_node 8 --master_port 29513 train_gen_varlen.py \
  --config configs/gpt_quadtree_3level.yaml \
  --shards "/tmp/eb_{0..7}.tar" \
  --max_tokens_global 1048576 --steps 100 --num_workers 4
echo "SMOKE_EXIT $?"
