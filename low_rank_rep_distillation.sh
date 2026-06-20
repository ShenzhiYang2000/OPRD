#!/bin/bash
# Low-rank cross-architecture representation distillation (Run C).
# Frozen teacher PCA bases P_T + trainable student projectors P_S in r-dim subspace.
# Delegates to on_policy_distillation.sh with low-rank defaults.
#
# Quick start (rep-only, r=256):
#   bash low_rank_rep_distillation.sh
#
# With pre-experiment 2 checkpoint:
#   REP_LOW_RANK_INIT_CHECKPOINT=outputs/cross_arch_preexp2/rank_256/ps_bank.pt \
#   bash low_rank_rep_distillation.sh
#
# With direct preexp2 checkpoint (joint trainable P_S/P_T, no teacher SVD):
#   REP_LOW_RANK_INIT_CHECKPOINT=outputs/cross_arch_preexp2_direct/rank_32_direct/direct_bank.pt \
#   bash low_rank_rep_distillation.sh
#
# With MLP preexp2 checkpoint (auto-detected from ps_bank.pt metadata):
#   REP_LOW_RANK_INIT_CHECKPOINT=outputs/cross_arch_preexp2/rank_256_mlp/ps_bank.pt \
#   REP_PS_PROJECTOR=auto bash low_rank_rep_distillation.sh
#
# OPD + low-rank rep:
#   REP_DISTILLATION_ONLY=False LOG_PROB_TOP_K=16 bash low_rank_rep_distillation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Rep distillation (same as rep_distillation.sh) ---
export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-False}
export REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-False}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-10.0}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_FIRST_K=${REP_DISTILLATION_FIRST_K:-2000}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}

# --- Low-rank cross-arch projector (Run C) ---
export REP_PROJECTOR_MODE=low_rank
export REP_LOW_RANK=${REP_LOW_RANK:-4}
# Pre-exp 2 init (optional). First training batch will estimate P_T via PCA if unset.
export REP_LOW_RANK_INIT_CHECKPOINT=${REP_LOW_RANK_INIT_CHECKPOINT:-${SCRIPT_DIR}/outputs/bridge_construction/rank_${REP_LOW_RANK}/ps_bank.pt}
# Freeze P_S from checkpoint; only student LLM is trained (requires ps_bank.pt for P_S).
export REP_FREEZE_PS=${REP_FREEZE_PS:-True}

# Attention distillation off by default for low-rank rep runs
export USE_ATT_DISTILLATION=${USE_ATT_DISTILLATION:-False}
export ATT_DISTILLATION_COEF=${ATT_DISTILLATION_COEF:-1000000.0}
export ATT_DISTILLATION_LAYERS=${ATT_DISTILLATION_LAYERS:-all}
export ATT_DISTILLATION_POSITIONS=${ATT_DISTILLATION_POSITIONS:-last_k}
export ATT_DISTILLATION_LAST_K=${ATT_DISTILLATION_LAST_K:-100}
export ATT_DISTILLATION_FIRST_K=${ATT_DISTILLATION_FIRST_K:-100}
export ATT_DISTILLATION_MAX_KEY_LEN=${ATT_DISTILLATION_MAX_KEY_LEN:-4096}
export ATT_DISTILLATION_LOSS=${ATT_DISTILLATION_LOSS:-mse}
export ATT_DISTILLATION_TEMPERATURE=${ATT_DISTILLATION_TEMPERATURE:-1.0}

# Rep-only by default (no OPD top-k). Set REP_DISTILLATION_ONLY=False LOG_PROB_TOP_K=16 for OPD+rep.
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}

export PROJECT_NAME=${PROJECT_NAME:-LowRankRepDistillation}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-oprd-bridge-r${REP_LOW_RANK}}

export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-Formal-DAPO-Math-17k-opd-top0_lr1e-5_Phi}

if [[ ! -f "${REP_LOW_RANK_INIT_CHECKPOINT}" ]]; then
  echo "NOTE: REP_LOW_RANK_INIT_CHECKPOINT not found: ${REP_LOW_RANK_INIT_CHECKPOINT}"
  echo "      P_T will be initialized from the first teacher batch (PCA). P_S starts random."
  unset REP_LOW_RANK_INIT_CHECKPOINT
fi

echo "Low-rank cross-OPRD: REP_PROJECTOR_MODE=${REP_PROJECTOR_MODE} REP_LOW_RANK=${REP_LOW_RANK} REP_FREEZE_PS=${REP_FREEZE_PS}"
if [[ -n "${REP_LOW_RANK_INIT_CHECKPOINT:-}" ]]; then
  echo "Init checkpoint: ${REP_LOW_RANK_INIT_CHECKPOINT}"
fi

exec bash "${SCRIPT_DIR}/on_policy_distillation.sh" "$@"
