#!/usr/bin/env bash
# aip main-script for the 512 2-level QuadTok tokenizer FULL training (H200, N nodes, 8 GPU/node).
# Mirrors pluto_launch_m3.sh: checkpoints on local SSD, continuously mirrored to S3; on (re)start
# pull the latest checkpoint from S3 so training survives localssd destruction + preemption.
# per_gpu_batch_size / num_pods (global batch 512) are decided from the smoke; set them via the
# aip submit (--num-pods) + the config (training.per_gpu_batch_size), or CLI overrides here.
set -uo pipefail
echo "=== quadtok 512 2-level tokenizer training ==="
date
nvidia-smi -L 2>/dev/null || true

REPO=/sensei-fs-3/users/yuchengm/code/quadtok/512
VENV=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv
PY="$VENV/bin/python"
LOCAL_OUT=/mnt/localssd/quadtok_512_out
S3_OUT=s3://g3i-data/yuchengm/quadtok_512/quadtok_512_2level_tokenizer_scratch
mkdir -p "$LOCAL_OUT"
cd "$REPO"

export GRPC_ENABLE_FORK_SUPPORT=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.cache/torch
export OMP_NUM_THREADS=8
if [ -z "${WANDB_API_KEY:-}" ]; then echo "[warn] WANDB_API_KEY not set -> WANDB offline"; export WANDB_MODE=offline; fi
export WANDB_PROJECT=quadtok_512

# ---- s5cmd ----
if ! command -v s5cmd >/dev/null 2>&1; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    | tar xz -C /tmp s5cmd 2>/dev/null && export PATH="/tmp:$PATH"
fi
command -v s5cmd >/dev/null 2>&1 && S5=1 || { S5=0; echo "[s3] no s5cmd -> S3 mirror DISABLED"; }
push_s3() { [ "$S5" = 1 ] && s5cmd sync "$LOCAL_OUT/" "$S3_OUT/" >/dev/null 2>&1 || true; }

# ---- resume: pull latest checkpoint (+ wandb_id) from S3 ----
if [ "$S5" = 1 ] && s5cmd ls "$S3_OUT/" >/dev/null 2>&1; then
  LATEST=$(s5cmd ls "$S3_OUT/" 2>/dev/null | grep -oE 'checkpoint-[0-9]+/' | tr -d '/' | sort -t- -k2 -n | tail -1)
  if [ -n "$LATEST" ]; then
    echo "[s3] resume: pulling $LATEST"
    mkdir -p "$LOCAL_OUT/$LATEST"; s5cmd cp "$S3_OUT/$LATEST/*" "$LOCAL_OUT/$LATEST/" 2>&1 | tail -2 || true
  fi
  s5cmd cp "$S3_OUT/wandb_id.txt" "$LOCAL_OUT/wandb_id.txt" 2>/dev/null || true
fi

# ---- background pusher: mirror localssd -> S3 every 5 min ----
( while true; do sleep 300; push_s3; done ) &
PUSHER=$!
trap 'kill $PUSHER 2>/dev/null || true' EXIT

# ---- multinode rendezvous (RunAI env) ----
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}
NNODES=${RUNAI_NUM_WORKERS:-1}
RANK=${RUNAI_NODE_RANK:-0}
MASTER=${RUNAI_MASTER_ADDR:-127.0.0.1}
PORT=${RUNAI_MASTER_PORT:-29500}
echo "launch: nnodes=$NNODES rank=$RANK master=$MASTER:$PORT gpus/node=$NGPU  out=$LOCAL_OUT mirror=$S3_OUT"

"$PY" -m accelerate.commands.launch \
  --mixed_precision=bf16 \
  --num_machines="$NNODES" \
  --num_processes="$((NNODES * NGPU))" \
  --machine_rank="$RANK" \
  --main_process_ip="$MASTER" \
  --main_process_port="$PORT" \
  --same_network \
  scripts/train_tokenizer.py \
    config=configs/training/single_stage/quadtok_ss512_vq_2level.yaml
rc=$?

kill $PUSHER 2>/dev/null || true
echo "[s3] final push"; push_s3
echo "=== 512 TRAIN EXIT $rc ==="
exit $rc
