#!/usr/bin/env bash
# 双卡 inference：5000 个 plan 分两半，卡0 跑 0-2499，卡1 跑 2500-4999，同时跑。
# 用法：bash plan_inference_2gpu.sh --config CONFIG --checkpoint CKPT --plans-dir DIR --output_dir OUT

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/inference/gpt_16k_base.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
PLANS_DIR="${PLANS_DIR:-tree_planning/plans}"
OUTPUT_DIR="${OUTPUT_DIR:-tree_planning/gen_out}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --plans-dir) PLANS_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: bash plan_inference_2gpu.sh --checkpoint /path [--config CONFIG] [--plans-dir tree_planning/plans] [--output_dir tree_planning/gen_out]"
  exit 1
fi

N=5000
HALF=$((N / 2))
echo "Dual-GPU: plans 0-$((HALF-1)) on GPU 0, $HALF-$((N-1)) on GPU 1"

CUDA_VISIBLE_DEVICES=0 python scripts/inference_generator_from_plan_pkl.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --plans-dir "$PLANS_DIR" --output_dir "$OUTPUT_DIR" \
  --start-idx 0 --end-idx $HALF "${EXTRA_ARGS[@]}" &

CUDA_VISIBLE_DEVICES=1 python scripts/inference_generator_from_plan_pkl.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --plans-dir "$PLANS_DIR" --output_dir "$OUTPUT_DIR" \
  --start-idx $HALF --end-idx $N "${EXTRA_ARGS[@]}" &

wait
echo "Both processes finished."
