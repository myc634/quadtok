#!/usr/bin/env bash
# aip main-script: 512 2-level FULL ImageNet pretokenization, 1 node / 8x A100, RESUMABLE.
# Uses the frozen 512 tokenizer EMA + guided tau=0.03 search; writes one output tar per input tar
# (pretok-train-XXXXX.tar) with skip-existing, so P2 preemption + auto-requeue continues cleanly.
set -uo pipefail
echo "=== quadtok 512 FULL pretokenization (8x A100, resumable) ==="; date
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8
REPO=/sensei-fs-3/users/yuchengm/code/quadtok/512
VENV=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv
PY="$VENV/bin/python"
cd "$REPO"
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.cache/torch HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 GRPC_ENABLE_FORK_SUPPORT=1
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}
OUT=/sensei-fs-3/users/yuchengm/data/quadtok_512_pretok
mkdir -p "$OUT"
echo "output -> $OUT   (already-done tars: $(ls "$OUT"/pretok-*.tar 2>/dev/null | wc -l) / 1470)"

"$PY" -m accelerate.commands.launch \
  --num_processes "${NGPU}" --num_machines 1 --mixed_precision bf16 --main_process_port 29570 \
  scripts/probe512_extract.py --tau 0.03 --bs 32 --hflip 1 --out "$OUT"

echo "=== PRETOK512 EXIT $? (tars now: $(ls "$OUT"/pretok-*.tar 2>/dev/null | wc -l) / 1470) ==="; date
