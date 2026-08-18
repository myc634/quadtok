#!/bin/bash
set -e
echo "=== gen train on $(hostname) $(date)  SIZE=${SIZE} GC=${GC} RUN=${RUN_NAME} ==="
# node rank from hostname <job>-<N>-<replica>; num nodes from RunAI WORLD_SIZE (=num pods)
NODE_RANK=$(hostname | sed -E 's/.*-([0-9]+)-[0-9]+$/\1/'); [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || NODE_RANK=0
NNODES=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC=$(( NNODES * 8 ))
echo "NODE_RANK=${NODE_RANK} NNODES=${NNODES} NPROC=${NPROC} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
(apt-get update >/dev/null 2>&1 && apt-get install -y python3.10-dev >/dev/null 2>&1) || echo "(dev skip)"

REPO=/sensei-fs-3/users/yuchengm/code/quadtok/quadtok-base-update
VENV=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv
cd "$REPO"
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if ! command -v s5cmd >/dev/null 2>&1; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    | tar xz -C /tmp s5cmd 2>/dev/null && export PATH=/tmp:$PATH || echo "(s5cmd fetch failed)"
fi

# pull pretok data -> localssd (25 GB, ~1-2 min; skip if already present)
DATA=/mnt/localssd/pretok
mkdir -p "$DATA"
HAVE=$(ls "$DATA"/train-*.tar 2>/dev/null | wc -l)
if [ "$HAVE" -lt 1470 ]; then
  echo "pulling pretok data -> ${DATA} (have ${HAVE}/1470) ..."; T=$(date +%s)
  s5cmd cp 's3://g3i-data/yuchengm/quadtok_pretok_2level_center_hflip/*' "$DATA/" 2>&1 | tail -1
  echo "pull done in $(($(date +%s)-T))s, tars=$(ls "$DATA"/train-*.tar 2>/dev/null | wc -l)"
fi

export SIZE MAX_TOKEN_GLOBAL=232448 LR=4e-4 END_LR=2e-5 WARMUP=50000 STEPS=400000 SAVE_EVERY=2500 WD=0.05 GC
export DATA_DIR="$DATA"
export CKPT_S3="s3://g3i-data/yuchengm/quadtok_gen_ckpt/${RUN_NAME}/"
export CKPT_LOCAL=/mnt/localssd/ckpt
export RUN_NAME
# public wandb (entity maoyucheng0321) -- optional monitoring
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=maoyucheng0321 WANDB_PROJECT=quadtok-gen-varlen
# WANDB_KEY passed via --env at submit (empty -> wandb off)

"$VENV/bin/python" -m accelerate.commands.launch \
  --num_processes "$NPROC" --num_machines "$NNODES" --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --mixed_precision bf16 \
  scripts/train_generator_varlen.py
echo "TRAIN_EXITED size=${SIZE} node=${NODE_RANK}"
