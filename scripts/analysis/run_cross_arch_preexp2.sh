#!/bin/bash
# Pre-experiment 2 launcher: train P_S with frozen teacher PCA bases P_T.
# Reuses on-policy pairs from pre-experiment 1.
#
# Modes (SUBSPACE_MODE):
#   full    - frozen teacher PCA P_T + trainable P_S (default)
#   direct  - trainable linear P_S and P_T jointly, no teacher SVD
#   residual - freeze head bridge, train tail on teacher residual

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" # 0,1,2,3
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}"
# TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_DIR}/Phi-4-mini-reasoning}"
RESPONSES_JSONL="${RESPONSES_JSONL:-${REPO_ROOT}/outputs/cross_arch_preexp1/on_policy_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/bridge_construction}"
SUBSPACE_MODE="${SUBSPACE_MODE:-full}"
POSITION_MODE="${POSITION_MODE:-last_k}"
LAST_K="${LAST_K:-20000}" #2000
FIRST_K="${FIRST_K:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-65536}"
MAX_PCA_ROWS="${MAX_PCA_ROWS:-16384}"
EPOCHS="${EPOCHS:-20}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LR="${LR:-1e-4}"
RANKS="${RANKS:-4}"
LAYER_MODE="${LAYER_MODE:-all}"
PROJECTOR="${PROJECTOR:-linear}"
MLP_HIDDEN_MULT="${MLP_HIDDEN_MULT:-4}"

python3 "${SCRIPT_DIR}/cross_arch_preexp2_train_ps.py" \
  --responses-jsonl "${RESPONSES_JSONL}" \
  --student-model-path "${STUDENT_MODEL_PATH}" \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --subspace-mode "${SUBSPACE_MODE}" \
  --position-mode "${POSITION_MODE}" \
  --last-k "${LAST_K}" \
  --first-k "${FIRST_K}" \
  --batch-size "${BATCH_SIZE}" \
  --max-batch-tokens "${MAX_BATCH_TOKENS}" \
  --max-pca-rows "${MAX_PCA_ROWS}" \
  --epochs "${EPOCHS}" \
  --eval-every "${EVAL_EVERY}" \
  --lr "${LR}" \
  --layer-mode "${LAYER_MODE}" \
  --projector "${PROJECTOR}" \
  --mlp-hidden-mult "${MLP_HIDDEN_MULT}" \
  --ranks ${RANKS} \
  ${COMPUTE_PROBE_COSINE:+--compute-probe-cosine}

echo "Done. See ${OUTPUT_DIR}/summary.json"
echo "  full:     rank_*/results.json (or rank_*_mlp/ for MLP)"
echo "  direct:   rank_*_direct/direct_bank.pt"
