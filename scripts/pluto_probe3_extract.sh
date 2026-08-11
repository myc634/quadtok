#!/usr/bin/env bash
# aip main-script: 3-level probing EXTRACT -> generator (M1) training data (A100 P1, 1 node/8 GPU).
# Runs probe3.py --mode extract over ImageNet-train wds shards (one shard per GPU), writing
# code_indices/lod_indices/patch_indices/cls tars to localssd, mirrored to S3.
#
# Env: T1, T2 (chosen thresholds), SHARD_START, SHARD_END (0..1469), CKPT (default latest),
#      S3TOK (tokenizer dir), S3OUT (pretokenized output dir). Multi-node = split shard range
#      across several jobs.
set -uo pipefail
REPO=/sensei-fs-3/users/yuchengm/code/quadtok/3level
PY=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/bin/python
S3TOK=${S3TOK:-s3://g3i-data/yuchengm/quadtok_3level/quadtok_3level_tokenizer_scratch}
T1=${T1:-0.05}; T2=${T2:-0.02}
SHARD_START=${SHARD_START:-0}; SHARD_END=${SHARD_END:-1469}
TRAIN=/sensei-fs-3/users/yuchengm/data/imagenet-wds/train
S3OUT=${S3OUT:-s3://g3i-data/yuchengm/quadtok_3level/pretokenized/t1_${T1}_t2_${T2}}
cd "$REPO"
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.torch HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
export GRPC_ENABLE_FORK_SUPPORT=1 TOKENIZERS_PARALLELISM=false
command -v s5cmd >/dev/null 2>&1 || { curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xz -C /tmp s5cmd; export PATH=/tmp:$PATH; }

CKPT=${CKPT:-$(s5cmd ls "$S3TOK/" 2>/dev/null | grep -oE 'checkpoint-[0-9]+/' | tr -d '/' | sort -t- -k2 -n | tail -1)}
echo "[extract] tokenizer=$S3TOK/$CKPT/ema_model  t1=$T1 t2=$T2  shards $SHARD_START..$SHARD_END -> $S3OUT"
CK=/mnt/localssd/m3_tok; mkdir -p "$CK/ema_model"
s5cmd cp "$S3TOK/$CKPT/ema_model/*" "$CK/ema_model/" >/dev/null 2>&1
OUT=/mnt/localssd/pretok; mkdir -p "$OUT"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-8}

i=0
for sh in $(seq "$SHARD_START" "$SHARD_END"); do
  g=$((i % NGPU)); shs=$(printf '%06d' "$sh")
  [ -f "$TRAIN/train-$shs.tar" ] || { echo "skip missing shard $shs"; continue; }
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/probe3.py --mode extract --tokenizer_weight "$CK" \
    --shards "$TRAIN/train-$shs.tar" --t1 "$T1" --t2 "$T2" \
    --output_tar "$OUT/train-$shs.tar" --batch_size 25 --num_workers 6 \
    > "$OUT/train-$shs.log" 2>&1 &
  i=$((i + 1))
  if [ $((i % NGPU)) -eq 0 ]; then
    wait
    s5cmd sync "$OUT/" "$S3OUT/" >/dev/null 2>&1 || true   # push finished shards, free localssd
    rm -f "$OUT"/train-*.tar
  fi
done
wait
s5cmd sync "$OUT/" "$S3OUT/" >/dev/null 2>&1 || true
echo "EXTRACT_ALL_DONE -> $S3OUT"
