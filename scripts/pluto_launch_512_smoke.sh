#!/usr/bin/env bash
# aip main-script for the 512 2-level QuadTok SMOKE: REAL training, 1 H200 node / 8 GPU, 500 steps.
# Goal: prove fwd/bwd stability at 512px, measure peak memory + throughput (to fix per_gpu / node
# count for the full run), and eyeball a step-500 reconstruction. NOT a paper result.
set -uo pipefail
echo "=== quadtok 512 2-level SMOKE (H200, 1 node, 8 GPU, 500 steps) ==="
date
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

REPO=/sensei-fs-3/users/yuchengm/code/quadtok/512
VENV=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv   # reuse the existing venv (same torch 2.7.1+cu128)
PY="$VENV/bin/python"                                       # call venv python directly (relocated-venv safe)
LOCAL_OUT=/mnt/localssd/quadtok_512_smoke
S3_OUT=s3://g3i-data/yuchengm/quadtok_512/smoke
mkdir -p "$LOCAL_OUT"
cd "$REPO"

export GRPC_ENABLE_FORK_SUPPORT=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.cache/torch
export OMP_NUM_THREADS=8
if [ -z "${WANDB_API_KEY:-}" ]; then echo "[warn] WANDB_API_KEY not set -> wandb offline"; export WANDB_MODE=offline; fi
export WANDB_PROJECT=quadtok_512

echo "--- python / torch / flex sanity ---"
"$PY" -c "import torch; from torch.nn.attention.flex_attention import flex_attention; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

# ---- s5cmd (for final log/sample mirror to S3; node localssd is ephemeral) ----
if ! command -v s5cmd >/dev/null 2>&1; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    | tar xz -C /tmp s5cmd 2>/dev/null && export PATH="/tmp:$PATH"
fi
command -v s5cmd >/dev/null 2>&1 && S5=1 || { S5=0; echo "[s3] no s5cmd -> S3 mirror disabled"; }

# ---- background peak-GPU-memory sampler ----
MEMLOG=/tmp/mem_peak.txt; echo 0 > "$MEMLOG"
( peak=0
  while true; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -n "$u" ] && [ "$u" -gt "$peak" ] 2>/dev/null && { peak=$u; echo "$peak" > "$MEMLOG"; }
    sleep 10
  done ) &
MEMSAMP=$!
trap 'kill $MEMSAMP 2>/dev/null || true' EXIT

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}
PER_GPU=${PER_GPU:-32}            # override at submit via --env PER_GPU=64
GRAD_CKPT=${GRAD_CKPT:-false}     # NOTE: not wired into the tokenizer's ResidualAttentionBlock yet
STEPS=${STEPS:-500}
echo "--- launch: ${STEPS}-step smoke on ${NGPU} GPUs, per_gpu=${PER_GPU}, grad_ckpt=${GRAD_CKPT}, global=$((NGPU*PER_GPU)) ---"
T0=$(date +%s)
"$PY" -m accelerate.commands.launch \
  --mixed_precision=bf16 \
  --num_machines=1 \
  --num_processes="${NGPU}" \
  --machine_rank=0 \
  --main_process_ip=127.0.0.1 \
  --main_process_port=29500 \
  --same_network \
  scripts/train_tokenizer.py \
    config=configs/training/single_stage/quadtok_ss512_vq_2level.yaml \
    experiment.name=quadtok_512_2level_smoke \
    experiment.output_dir="$LOCAL_OUT" \
    experiment.use_wandb_id=false \
    experiment.resume=false \
    experiment.save_every=100000 \
    experiment.generate_every=100000 \
    training.enable_wandb=false \
    training.max_train_steps=$STEPS \
    training.per_gpu_batch_size=$PER_GPU \
    model.grad_checkpointing=$GRAD_CKPT
rc=$?
T1=$(date +%s)

kill $MEMSAMP 2>/dev/null || true
echo ""
echo "=== SMOKE SUMMARY ==="
echo "exit_code           : $rc"
echo "wall_time_seconds   : $((T1 - T0))  (includes model build + data warmup)"
echo "peak_gpu_mem_MiB    : $(cat $MEMLOG 2>/dev/null)  (per-GPU max, H200=143GB=146304MiB)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
echo "--- localssd out ---"; ls -R "$LOCAL_OUT" 2>/dev/null | head -40

if [ "$S5" = 1 ]; then
  echo "[s3] mirroring smoke logs + samples -> $S3_OUT"
  s5cmd sync "$LOCAL_OUT/" "$S3_OUT/" >/dev/null 2>&1 || true
fi
echo "=== 512 SMOKE EXIT $rc ==="
exit $rc
