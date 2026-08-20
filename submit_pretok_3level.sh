#!/usr/bin/env bash
# Self-submitting AIP job: full-ImageNet-train 3-level pretokenization on 1 node (8x A100, P2).
# RESUMABLE: work is split into CHUNK-shard tars written to sensei-fs; a chunk with a .done marker
# is skipped, so preemption + auto-recovery just continues. Aug = center-crop(BICUBIC)+hflip (2
# views/img); order = generator BFS (SLOT_RANK); codes int16/int8. Weight pre-staged on sensei-fs.
#
#   submit:  bash submit_pretok_3level.sh
#   (pod re-runs this with JOB_UUID set -> run_remote)
set -uo pipefail
AIP_BIN="${AIP_BIN:-$HOME/venv/pluto/bin/aip}"
JOB_NAME="${JOB_NAME:-pretok-3level-a100p2}"
PROJECT="${PROJECT:-SceneStaging}"
IMAGE="${IMAGE:-docker-matrix-experiments-snapshot.ff.adobe.io/training-base-07-23-25:12.8.1-runtime-ubuntu22.04-CUDNN-v9.11.0.98-NCCL-v2.27.3-1}"
GPU_INSTANCE_TYPE="${GPU_INSTANCE_TYPE:-p4de.24xlarge}"   # A100-80G x8
QUOTA_FLAG="${QUOTA_FLAG:---preemptible}"                  # P2
XPUS_PER_POD="${XPUS_PER_POD:-8}"; NUM_PODS="${NUM_PODS:-1}"

REPO_FS="${REPO_FS:-/sensei-fs-3/users/yuchengm/code/quadtok/3level}"
PY310="${PY310:-/sensei-fs-3/users/yuchengm/.uvpy/cpython-3.10-linux-x86_64-gnu/bin/python3.10}"
SITEPKG="${SITEPKG:-/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/lib/python3.10/site-packages}"
WEIGHT="${WEIGHT:-/sensei-fs-3/users/yuchengm/data/quadtok_hf_weight/Tokenizer/3level/checkpoint-350000}"
OUT="${OUT:-/sensei-fs-3/users/yuchengm/data/quadtok_pretok_3level}"
TRAIN_PREFIX="${TRAIN_PREFIX:-/sensei-fs-3/users/yuchengm/data/imagenet-wds/train/train}"
NSHARDS="${NSHARDS:-1470}"; CHUNK="${CHUNK:-10}"
T1="${T1:-0.004}"; T2="${T2:-0.021}"; BS="${BS:-64}"; NW="${NW:-8}"

run_aip() { "$AIP_BIN" "$@"; }

run_remote() {
  export PYTHONPATH="$SITEPKG"
  export TORCH_HOME=/sensei-fs-3/users/yuchengm/.torch HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false GRPC_ENABLE_FORK_SUPPORT=1
  [ -f "$WEIGHT/ema_model/pytorch_model.bin" ] || { echo "WEIGHT MISSING: $WEIGHT" >&2; exit 1; }
  mkdir -p "$OUT"; cd "$REPO_FS"
  NCHUNK=$(( (NSHARDS + CHUNK - 1) / CHUNK ))
  echo "pretok start: NSHARDS=$NSHARDS CHUNK=$CHUNK -> $NCHUNK chunks, 8 GPUs, hflip, int16 -> $OUT"
  for g in 0 1 2 3 4 5 6 7; do
    (
      for (( c=g; c<NCHUNK; c+=8 )); do
        lo=$(( c*CHUNK )); hi=$(( lo+CHUNK-1 )); [ $hi -ge $NSHARDS ] && hi=$(( NSHARDS-1 ))
        out="$OUT/pretok-c$(printf '%04d' $c).tar"
        [ -f "$out.done" ] && { echo "skip c$c (done)"; continue; }
        SH=$(printf "%s-{%06d..%06d}.tar" "$TRAIN_PREFIX" $lo $hi)
        CUDA_VISIBLE_DEVICES=$g "$PY310" scripts/search3_fast.py --mode extract --weight "$WEIGHT" \
          --shards "$SH" --t1 $T1 --t2 $T2 --hflip 1 --bs $BS --num_workers $NW --limit 0 \
          --output_tar "$out" > "$OUT/log_g${g}_c${c}.log" 2>&1 && touch "$out.done" \
          && echo "done c$c (g$g, shards $lo..$hi)" || echo "FAIL c$c (g$g) -- see $OUT/log_g${g}_c${c}.log"
      done
    ) &
  done
  wait
  echo "ALL_CHUNKS_ATTEMPTED: $(ls "$OUT"/*.tar.done 2>/dev/null | wc -l) / $NCHUNK done, $(ls "$OUT"/*.tar 2>/dev/null | wc -l) tars"
}

submit_job() {
  run_aip job create --name "$JOB_NAME" --project "$PROJECT" --job-type training $QUOTA_FLAG \
    --gpu-instance-type "$GPU_INSTANCE_TYPE" --xpus-per-pod "$XPUS_PER_POD" --num-pods "$NUM_PODS" \
    --image "$IMAGE" --main-script "$0"
  sleep 10
  run_aip job start "$JOB_NAME"
}

if [ -n "${JOB_UUID:-}" ]; then run_remote; else submit_job; fi
