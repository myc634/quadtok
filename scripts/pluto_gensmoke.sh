#!/usr/bin/env bash
# aip main-script: 512 QuadtreeGPT generator VARLEN smoke, 1 H200 node / 8 GPU, 100 steps.
# per-GPU token budget = global_max_tokens(262144*4) // 8 = 131072. Verifies the generator trains
# on 8xH200 with varlen flash-attn + no padding.
set -uo pipefail
echo "=== quadtok 512 generator varlen smoke (8x H200, 100 steps) ==="; date
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8
REPO=/sensei-fs-3/users/yuchengm/code/quadtok/512
PY=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/bin/python
cd "$REPO"
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.cache/torch HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 GRPC_ENABLE_FORK_SUPPORT=1
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}
echo "--- accelerate launch on ${NGPU} GPUs ---"
"$PY" -m accelerate.commands.launch \
  --num_processes "${NGPU}" --num_machines 1 --mixed_precision bf16 --main_process_port 29543 \
  scripts/smoke_gen512.py --steps 100
echo "=== GENSMOKE8 EXIT $? ==="
