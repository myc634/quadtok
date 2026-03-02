#!/usr/bin/env bash
# 在 quadtok_tree_plan_vllm 环境下启动 vLLM serve。
# 1) libcudart.so.12：用 nvidia-cuda-runtime-cu12 提供的库路径。
# 2) CXXABI_1.3.15：优先用 conda 的 lib（libstdc++、libicu 等），避免系统旧 libstdc++ 导致 ImportError。
#
# 3) Qwen3.5 + v1 引擎：CUDA graph 捕获时 causal_conv1d 会触发 num_cache_lines >= batch 断言，
#    加 --enforce-eager 禁用 CUDA graph 可规避（会稍慢，但能正常跑）。
#
# 用法（二选一）：
#   1) conda activate quadtok_tree_plan_vllm && ./run_vllm_serve.sh [参数...]
#   2) conda run -n quadtok_tree_plan_vllm ./run_vllm_serve.sh [参数...]
# 示例：./run_vllm_serve.sh Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 2 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only --enforce-eager

set -e
CONDA_ENV=quadtok_tree_plan_vllm
CONDA_ROOT="${CONDA_PREFIX:-/home/ubuntu/project-quadtok/miniforge3/envs/${CONDA_ENV}}"
# conda lib 放最前：提供 libstdc++.so.6（含 CXXABI_1.3.15），满足 libicui18n 等依赖
CONDA_LIB="${CONDA_ROOT}/lib"
CU12_LIB="${CONDA_ROOT}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CU12_LIB}:${LD_LIBRARY_PATH:-}"

vllm serve Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 2 --max-model-len 1024 --reasoning-parser qwen3 --language-model-only
