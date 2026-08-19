#!/bin/bash
# ============================================================================
# 1-node 8x B200 generator training (QuadtreeGPT xlarge / 700m).
# GOAL: keep GLOBAL token budget FIXED, fully utilize 1-node B200, train fastest.
#
# Why this is the fastest config at a FIXED global batch on 1 node:
#   * use ALL 8 GPUs  -> per-GPU work = MAX_TOKEN_GLOBAL/8 is smallest -> fastest
#     wall-clock/step (data-parallel). 8x B200 ~= 2x faster than 4x for same global batch.
#   * GC=0 (grad checkpointing OFF) -> B200's 180-192GB has plenty of room, so we
#     skip activation recompute => ~25-30% faster backward. (GC was only ever ON
#     because A100/H100 are 80GB.)
#   * grad-accum = 1 (never used) -> global batch comes from the token budget, not accumulation.
#   * bf16 + flash-attn varlen (no padding). torch.compile gave no gain on H100;
#     optional to re-test on Blackwell.
#   NOTE: per-GPU = 29056 tokens (232448/8) uses only ~20GB of 180GB. That is FINE:
#   the light per-GPU load is exactly why wall-clock is fast. At a fixed global batch
#   you cannot convert spare memory into more speed without changing global batch or GPU count.
# ============================================================================
set -e

export SIZE=${SIZE:-xlarge}                                # 700m
export GC=${GC:-0}                                         # B200: grad checkpointing OFF (the whole point)
export MAX_TOKEN_GLOBAL=${MAX_TOKEN_GLOBAL:-232448}        # FIXED global batch (~1024 imgs). Keep it to stay comparable.
export LR=${LR:-4e-4} END_LR=${END_LR:-2e-5} WARMUP=${WARMUP:-50000}
export STEPS=${STEPS:-400000} SAVE_EVERY=${SAVE_EVERY:-2500} WD=${WD:-0.05}
export NUM_WORKERS=${NUM_WORKERS:-12}                      # B200 compute is fast -> feed it (bump dataloader workers)
export DATA_DIR=${DATA_DIR:-/data/quadtok/pretok_2level}   # local NVMe (pull from HF yuchengm/quadtok_data first)
export CKPT_LOCAL=${CKPT_LOCAL:-/data/ckpt_local}
export RUN_NAME=${RUN_NAME:-gen_700m_b200}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ckpt read(resume)+write location. FROM SCRATCH -> point to a NEW empty path (no step_N).
#                                   RESUME       -> point to an existing dir that has step_N.
: "${CKPT_S3:?set CKPT_S3 (new empty path = from scratch, or existing gen_700m/ = resume). If no S3, see TRAIN_700M.md 'de-S3'.}"
# optional wandb (same run on resume via id=RUN_NAME)
export WANDB_ENTITY=${WANDB_ENTITY:-} WANDB_PROJECT=${WANDB_PROJECT:-quadtok-gen-varlen}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
echo "[b200] SIZE=$SIZE GC=$GC global_tok=$MAX_TOKEN_GLOBAL per_gpu=$((MAX_TOKEN_GLOBAL/8)) run=$RUN_NAME"
accelerate launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
  scripts/train_generator_varlen.py
