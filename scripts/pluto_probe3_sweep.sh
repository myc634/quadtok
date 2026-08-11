#!/usr/bin/env bash
# aip main-script: 3-level probing threshold SWEEP (A100 P1, 1 node / 8 GPU).
# Runs probe3.py --mode eval over a (t1,t2) grid, one combo per GPU (rounds if >NGPU combos),
# on 10k ImageNet-val images. Goal: pick the (t1,t2) with mean tokens ~800-900 and best
# rFID/PSNR/LPIPS. RESULT_JSON per combo -> printed + synced to S3.
#
# Env overrides: COMBOS ("t1:t2 ..."), LIMIT (default 10000), CKPT (checkpoint-N; default latest),
#                S3TOK (tokenizer S3 dir), S3OUT (results S3 dir).
set -uo pipefail
REPO=/sensei-fs-3/users/yuchengm/code/quadtok/3level
PY=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/bin/python
S3TOK=${S3TOK:-s3://g3i-data/yuchengm/quadtok_3level/quadtok_3level_tokenizer_scratch}
COMBOS=${COMBOS:-"0.02:0.005 0.03:0.01 0.05:0.01 0.03:0.02 0.05:0.02 0.05:0.03 0.08:0.02 0.08:0.03"}
LIMIT=${LIMIT:-10000}
S3OUT=${S3OUT:-s3://g3i-data/yuchengm/quadtok_3level/probe3_sweep}
VAL="/sensei-fs-3/users/yuchengm/data/imagenet-wds/val/val-{000000..000049}.tar"
cd "$REPO"
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.torch HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export GRPC_ENABLE_FORK_SUPPORT=1 TOKENIZERS_PARALLELISM=false
command -v s5cmd >/dev/null 2>&1 || { curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz -C /tmp s5cmd; export PATH=/tmp:$PATH; }

CKPT=${CKPT:-$(s5cmd ls "$S3TOK/" 2>/dev/null | grep -oE 'checkpoint-[0-9]+/' | tr -d '/' | sort -t- -k2 -n | tail -1)}
echo "[sweep] tokenizer = $S3TOK/$CKPT/ema_model ; combos = $COMBOS ; limit=$LIMIT"
CK=/mnt/localssd/m3_tok; mkdir -p "$CK/ema_model"
s5cmd cp "$S3TOK/$CKPT/ema_model/*" "$CK/ema_model/" >/dev/null 2>&1
OUT=/mnt/localssd/sweep_out; mkdir -p "$OUT"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}

i=0
for combo in $COMBOS; do
  t1=${combo%%:*}; t2=${combo##*:}; g=$((i % NGPU)); tag="t1_${t1}_t2_${t2}"
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/probe3.py --mode eval --tokenizer_weight "$CK" \
    --shards "$VAL" --t1 "$t1" --t2 "$t2" --limit "$LIMIT" --batch_size 25 --num_workers 6 \
    --output "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1 &
  i=$((i + 1))
  [ $((i % NGPU)) -eq 0 ] && wait
done
wait
echo "=== SWEEP RESULTS (mean tokens ~800-900 + best rFID/PSNR/LPIPS) ==="
grep -h RESULT_JSON "$OUT"/*.log 2>/dev/null
s5cmd sync "$OUT/" "$S3OUT/" >/dev/null 2>&1 || true
echo "SWEEP_DONE ($S3OUT)"
