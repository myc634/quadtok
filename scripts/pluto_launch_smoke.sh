#!/usr/bin/env bash
# aip main-script for the 3-level QuadTok SMOKE (A100 P1, 1 node, uses 1 GPU).
set -uo pipefail
echo "=== quadtok 3-level SMOKE (A100 P1) ==="
date
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

REPO=/sensei-fs-3/users/yuchengm/code/quadtok/3level
VENV=/sensei-fs-3/users/yuchengm/code/quadtok/.venv
cd "$REPO"
source "$VENV/bin/activate"

export GRPC_ENABLE_FORK_SUPPORT=1
export CUDA_VISIBLE_DEVICES=0            # smoke needs only 1 GPU of the node
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.cache/torch

echo "--- python / torch / flex sanity ---"
python -c "import torch; from torch.nn.attention.flex_attention import flex_attention; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "--- run smoke ---"
python scripts/smoke_test_3level.py config=configs/training/single_stage/quadtok_ss256_vq_3level.yaml
rc=$?
echo "=== SMOKE EXIT $rc ==="
exit $rc
