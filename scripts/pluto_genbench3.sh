#!/usr/bin/env bash
# aip main-script: 512 QuadtreeGPT varlen generator PARITY + SPEED SWEEP, 1 H200 node / 8 GPU.
# Order matches the b->c plan:
#   (b) no-pad vectorized varlen embedding  (removes ~128 host-side CPU-GPU syncs/step)
#   (c) Liger fused-linear-CE head          (no ~8.6GB fp32 logits -> may unlock a higher ckpt interval)
# First runs parity (padded forward == varlen-vanilla == varlen-liger) as a GATE, then benches the
# interval x compile x use_liger x grad-accum grid at the real 131072 tok/GPU budget. Each config is a
# fresh accelerate process (clean compile cache + peak-mem). One RESULT line per config.
set -uo pipefail
echo "=== quadtok 512 generator PARITY + SPEED SWEEP (8x H200) ==="; date
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8
REPO=/sensei-fs-3/users/yuchengm/code/quadtok/512
VENV=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv
PY="$VENV/bin/python"
cd "$REPO"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.cache/torch HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 GRPC_ENABLE_FORK_SUPPORT=1
export TORCHINDUCTOR_CACHE_DIR=/sensei-fs-3/users/yuchengm/.cache/inductor_gen512
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}

echo ""; echo "=== (c) ensure liger-kernel in venv (triton comes from torch; --no-deps avoids torch churn) ==="
"$PY" -c "import triton; print('triton', triton.__version__)" || { echo "NO TRITON - liger unsafe"; }
"$PY" -c "from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss" 2>/dev/null \
  || uv pip install --python "$PY" --no-deps liger-kernel 2>&1 | tail -6
"$PY" -c "from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; print('liger FLCE import OK')" \
  || echo "LIGER IMPORT FAILED (use_liger legs will fall back / error)"

echo ""; echo "=== PARITY GATE (b + c): padded forward vs varlen-vanilla vs varlen-liger ==="
"$PY" scripts/parity_gen512.py 2>&1 | tee /tmp/parity.log
if grep -q "FAIL" /tmp/parity.log; then
  echo "!!! PARITY FAILED — numbers below would be meaningless. Aborting sweep."; echo "=== GENBENCH3 EXIT 1 ==="; date; exit 1
fi
echo "PARITY OK -> proceeding to speed sweep"

run() {  # $1 ckpt_interval  $2 compile  $3 use_liger  $4 accum  $5 port
  echo ""; echo ">>> BENCH ckpt_interval=$1 compile=$2 use_liger=$3 accum=$4 <<<"
  "$PY" -m accelerate.commands.launch \
    --num_processes "${NGPU}" --num_machines 1 --mixed_precision bf16 --main_process_port "$5" \
    scripts/bench_gen512.py --ckpt_interval "$1" --compile "$2" --use_liger "$3" --accum "$4" \
    --steps 30 --warmup 12 \
    2>&1 | grep --line-buffered -E "^\[cfg\]|^  optstep|^RESULT|Error|OutOfMemory|Traceback|liger|no_sync" || true
}

# (b) baseline: production config (interval2 + compile) with the new vectorized no-pad embedding,
#     vanilla head -> compare s/step to the handoff's 1.24 (old host-loop) to confirm (b) didn't regress.
run 2 1 0 1 29551
# (c) liger on the same operating point: expect large peak-mem drop, s/step ~same-or-better.
run 2 1 1 1 29552
# (c) does liger's freed ~22GB unlock fewer recomputes (faster)?  interval 3 / 4.
run 3 1 1 1 29553
run 4 1 1 1 29554
# grad-accum at the best low-ckpt point (throughput invariant; checks util + opt/comm overhead).
run 3 1 1 2 29555
# no-ckpt LAST: it OOMs at 131k tok/GPU even with liger (~138GB > 140), and a hard OOM leaves the CUDA/
# NCCL context dirty -> it would poison any config run after it. Keep OOM-prone configs at the end.
run 0 1 1 1 29556

echo ""; echo "=== GENBENCH3 EXIT $? ==="; date
