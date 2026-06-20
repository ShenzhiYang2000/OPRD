#!/bin/bash
# Pre-experiment 1 + 3 launcher (single GPU).
# Edit model paths before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}"
DATA_PARQUET="${DATA_PARQUET:-${REPO_ROOT}/datasets/dapo-math-17k.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/cross_arch_preexp1}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
LAST_K="${LAST_K:-2000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
GENERATE_BACKEND="${GENERATE_BACKEND:-vllm}"
GENERATE_BATCH_SIZE="${GENERATE_BATCH_SIZE:-16}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-8}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-131072}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-0}"

# Reuse cached on-policy pairs if already generated:
# RESPONSES_JSONL="${OUTPUT_DIR}/on_policy_pairs.jsonl"

python3 "${SCRIPT_DIR}/cross_arch_repr_analysis.py" \
  --student-model-path "${STUDENT_MODEL_PATH}" \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --data-parquet "${DATA_PARQUET}" \
  --num-prompts "${NUM_PROMPTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --last-k "${LAST_K}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generate-backend "${GENERATE_BACKEND}" \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --batch-size "${FORWARD_BATCH_SIZE}" \
  --max-batch-tokens "${MAX_BATCH_TOKENS}" \
  --vllm-tensor-parallel-size "${VLLM_TP}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEM}" \
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --generate-responses \
  ${RESPONSES_JSONL:+--responses-jsonl "${RESPONSES_JSONL}"}

echo "Done. See ${OUTPUT_DIR}/results.json and figures."
