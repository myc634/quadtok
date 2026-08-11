#!/usr/bin/env bash
# aip main-script for M3: 3-level QuadTok tokenizer training (H200 P2, 1 node, 8 GPU).
#
# Checkpoints live on LOCAL SSD (fast, ephemeral, off the 500GB sensei-fs quota) and are
# continuously mirrored to S3; on (re)start we pull the latest checkpoint back from S3 so
# training survives localssd destruction + preemption. auto_resume (resume:True) then finds
# the pulled checkpoint locally and continues; use_wandb_id continues the same wandb run.
set -uo pipefail
echo "=== quadtok 3-level M3 tokenizer training ==="
date
nvidia-smi -L 2>/dev/null || true

REPO=/sensei-fs-3/users/yuchengm/code/quadtok/3level
VENV=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv
PY="$VENV/bin/python"            # call the venv python directly: the venv was relocated to
                                 # base/.venv, so its activate/console-scripts (accelerate,
                                 # wandb) have stale shebangs -> use `python -m` instead.
LOCAL_OUT=/mnt/localssd/quadtok_3level_out
S3_OUT=s3://g3i-data/yuchengm/quadtok_3level/quadtok_3level_tokenizer_scratch
mkdir -p "$LOCAL_OUT"
cd "$REPO"

export GRPC_ENABLE_FORK_SUPPORT=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.cache/torch
export OMP_NUM_THREADS=8
if [ -z "${WANDB_API_KEY:-}" ]; then echo "[warn] WANDB_API_KEY not set -> WANDB offline"; export WANDB_MODE=offline; fi
export WANDB_PROJECT=quadtok_3level

# ---- ensure s5cmd (static binary; install to /tmp if the image lacks it) ----
if ! command -v s5cmd >/dev/null 2>&1; then
  echo "[s3] s5cmd not found, installing to /tmp"
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    | tar xz -C /tmp s5cmd 2>/dev/null && export PATH="/tmp:$PATH"
fi
if command -v s5cmd >/dev/null 2>&1; then S5=1; else S5=0; echo "[s3] WARNING: no s5cmd -> S3 mirroring DISABLED"; fi

push_s3() { [ "$S5" = 1 ] && s5cmd sync "$LOCAL_OUT/" "$S3_OUT/" >/dev/null 2>&1 || true; }

# ---- resume: pull the latest checkpoint (+ wandb_id) from S3 into localssd ----
if [ "$S5" = 1 ] && s5cmd ls "$S3_OUT/" >/dev/null 2>&1; then
  LATEST=$(s5cmd ls "$S3_OUT/" 2>/dev/null | grep -oE 'checkpoint-[0-9]+/' | tr -d '/' | sort -t- -k2 -n | tail -1)
  if [ -n "$LATEST" ]; then
    echo "[s3] resume: pulling $LATEST from S3"
    mkdir -p "$LOCAL_OUT/$LATEST"
    s5cmd cp "$S3_OUT/$LATEST/*" "$LOCAL_OUT/$LATEST/" 2>&1 | tail -2 || true
  fi
  s5cmd cp "$S3_OUT/wandb_id.txt" "$LOCAL_OUT/wandb_id.txt" 2>/dev/null || true
  echo "[s3] local checkpoints after pull:"; ls "$LOCAL_OUT" 2>/dev/null | grep -E 'checkpoint|wandb_id' || echo "(none -> training from scratch)"
fi

# ---- background pusher: mirror localssd -> S3 every 5 min ----
( while true; do sleep 300; push_s3; done ) &
PUSHER=$!
trap 'kill $PUSHER 2>/dev/null || true' EXIT

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}
echo "launching accelerate (python -m) on ${NGPU} GPUs; out=$LOCAL_OUT  mirror=$S3_OUT"
"$PY" -m accelerate.commands.launch \
  --mixed_precision=bf16 \
  --num_machines=1 \
  --num_processes="${NGPU}" \
  --machine_rank=0 \
  --main_process_ip=127.0.0.1 \
  --main_process_port=29500 \
  --same_network \
  scripts/train_tokenizer.py \
    config=configs/training/single_stage/quadtok_ss256_vq_3level.yaml
rc=$?

kill $PUSHER 2>/dev/null || true
echo "[s3] final push localssd -> S3"; push_s3
echo "=== M3 TRAIN EXIT $rc ==="
exit $rc
