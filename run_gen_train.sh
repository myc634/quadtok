#!/bin/bash
# Real 3-level QuadtreeGPT (344M) training on ONE 8xH200 node — FASTEST measured recipe (2.13x
# vs ckpt-every-layer + eager): torch.compile + NO grad-checkpointing @ 65,536 tok/GPU microbatch
# + grad_accum 2  =>  effective global batch 1,048,576 tok (~1024 imgs), 1.00 s/opt-step, 90 GB.
#
# Set SHARDS to the PERSISTENT pretokenized ImageNet-train tars (see GENERATION_TRAINING_INFRA.md
# section 1; do NOT use pod-local /tmp for the real run).
PY310=/sensei-fs-3/users/yuchengm/.uvpy/cpython-3.10-linux-x86_64-gnu/bin/python3.10
export PYTHONPATH=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/lib/python3.10/site-packages
# On the training-base python3.10 image you can instead use base/.venv/bin/python + its torchrun.
cd /sensei-fs-3/users/yuchengm/code/quadtok/3level
export OMP_NUM_THREADS=8
SHARDS="${SHARDS:?set SHARDS=/sensei-fs-3/.../pretok-{000000..NNNNNN}.tar (persistent pretokenized data)}"
STEPS="${STEPS:-300000}"

$PY310 -m torch.distributed.run --nproc_per_node 8 --master_port 29531 train_gen_varlen.py \
  --config configs/gpt_quadtree_3level.yaml \
  --shards "$SHARDS" \
  --max_tokens_global 524288 \
  --grad_accum 2 \
  --ckpt_every 0 \
  --compile \
  --steps "$STEPS" \
  --lr 4e-4 \
  --num_workers 4
echo "TRAIN_EXIT $?"
