#!/bin/bash
# Representation-level distillation launcher.
# Sets rep-distillation defaults and delegates to on_policy_distillation.sh.
# ulimit -c 0


export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-True}
export REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-True}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}
# Token positions for rep loss: last | all | last_k | first_k
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_FIRST_K=${REP_DISTILLATION_FIRST_K:-2000}
# Transformer layers for rep loss: last | all | even | odd
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}

# Cross-arch rep projector: full (naive Linear d_S->d_T) | low_rank (frozen P_T + trainable P_S)
export REP_PROJECTOR_MODE=${REP_PROJECTOR_MODE:-full}
export REP_LOW_RANK=${REP_LOW_RANK:-256}
# Optional: outputs/cross_arch_preexp2/rank_256/ps_bank.pt
# export REP_LOW_RANK_INIT_CHECKPOINT=/path/to/ps_bank.pt

# Attention distillation (optional, can combine with rep / OPD)
# export USE_ATT_DISTILLATION=${USE_ATT_DISTILLATION:-False}
# export ATT_DISTILLATION_COEF=${ATT_DISTILLATION_COEF:-1.0}
# export ATT_DISTILLATION_LAYERS=${ATT_DISTILLATION_LAYERS:-all}
# export ATT_DISTILLATION_POSITIONS=${ATT_DISTILLATION_POSITIONS:-last_k}
# export ATT_DISTILLATION_LAST_K=${ATT_DISTILLATION_LAST_K:-100}
# export ATT_DISTILLATION_FIRST_K=${ATT_DISTILLATION_FIRST_K:-100}
# export ATT_DISTILLATION_MAX_KEY_LEN=${ATT_DISTILLATION_MAX_KEY_LEN:-4096}
# export ATT_DISTILLATION_LOSS=${ATT_DISTILLATION_LOSS:-mse}
# export ATT_DISTILLATION_TEMPERATURE=${ATT_DISTILLATION_TEMPERATURE:-1.0}

# OPD+rep (default): REP_DISTILLATION_ONLY=False, LOG_PROB_TOP_K=16
# Rep-only: REP_DISTILLATION_ONLY=True LOG_PROB_TOP_K=0
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}

export PROJECT_NAME=${PROJECT_NAME:-RepDistillation_logp}

# export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-opd_top0-rep_all_last25-att_all_last25}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-opd-rep_all_last2000_coef1-TEST-peak-memory}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"












# # All layers + all response tokens
# REP_DISTILLATION_LAYERS=all REP_DISTILLATION_POSITIONS=all bash rep_distillation.sh

# # Even layers + last token
# REP_DISTILLATION_LAYERS=even REP_DISTILLATION_POSITIONS=last bash rep_distillation.sh

# # Odd layers
# REP_DISTILLATION_LAYERS=odd bash rep_distillation.sh