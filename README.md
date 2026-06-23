# OPRD: On-Policy Representation Distillation

**Representation-level on-policy distillation for large language models, built on the [OPD](https://github.com/thunlp/OPD) training stack.**

---

## 🔔 News

- **[2026/06/22]** New arXiv update! The latest version of our paper on cross-architecture and cross-vocabulary representation distillation is now live! Check it out here: https://arxiv.org/abs/2606.06021
- **[2026/06/20]** We now support **cross-architecture** and **cross-tokenizer** distillation via OPRD-Bridge (e.g., Phi-4-mini-reasoning -> Qwen3-1.7B with completely different tokenizers).
- **[2026/06/04]** Paper released: [OPRD: On-Policy Representation Distillation](https://arxiv.org/abs/2606.06021).

---

[Paper (OPRD)](https://arxiv.org/abs/2606.06021)
[Upstream OPD](https://github.com/thunlp/OPD)
[Paper (OPD)](https://arxiv.org/abs/2604.13016)
[verl](https://github.com/verl-project/verl)

**[Overview](#overview)** | **[Method](#method)** | **[Getting Started](#getting-started)** | **[Memory](#memory-profiling)** | **[Validation](#validation)** | **[Citation](#citation)**

---

## Overview

**OPRD** distills a **teacher** into a **student** during on-policy rollouts by matching **hidden representations** on student-generated responses, instead of (or in addition to) matching token-level log-probabilities over a large vocabulary.

This repository implements two variants:

- **OPRD-Vanilla**: Same-architecture distillation (e.g., Qwen3-4B -> Qwen3-1.7B). Aligns hidden states directly via proportional layer mapping.
- **OPRD-Bridge**: Cross-architecture and cross-tokenizer distillation (e.g., Phi-4-mini-reasoning -> Qwen3-1.7B). Constructs a frozen low-rank bridge between heterogeneous teacher and student representation spaces via PCA + learned projectors.

Compared to top-*k* token OPD on long chain-of-thought responses, OPRD:

- Provides a **deterministic per-sample gradient**, removing the token-level estimation variance of OPD
- Avoids materializing the full-vocabulary `[B, T, |V|]` logit tensor during the actor update (**1.44x faster**, up to **54% less transient memory**)
- Supports **multi-layer** alignment with proportional layer indexing
- Enables **cross-architecture** and **cross-tokenizer** distillation (via OPRD-Bridge)

---

## Method

### OPRD-Vanilla (same architecture)

1. **Rollout**: The student generates on-policy responses (same pipeline as OPD).
2. **Teacher cache**: The teacher runs a forward pass; per-layer hidden states on the response region are stored.
3. **Student update**: The student forward produces matching hidden states; the loss is **MSE** between student and teacher representations.

Key knobs (see `rep_distillation.sh`):

| Concept             | Env vars                                  | Options                                             |
| ------------------- | ----------------------------------------- | --------------------------------------------------- |
| Token positions     | `REP_DISTILLATION_POSITIONS`              | `last`, `all`, `last_k`, `first_k`                  |
| Layers              | `REP_DISTILLATION_LAYERS`                 | `last`, `all`, `even`, `odd`                        |
| Rep-only vs OPD+rep | `REP_DISTILLATION_ONLY`, `LOG_PROB_TOP_K` | `True` + `0` = rep-only; `False` + `K>0` = combined |

### OPRD-Bridge (cross-architecture / cross-tokenizer)

When teacher and student have different hidden dimensions, depths, or even different tokenizers, direct hidden-state alignment fails. OPRD-Bridge addresses this via a two-stage pipeline:

**Stage 0: On-Policy Pair Collection** (`scripts/analysis/run_cross_arch_analysis.sh`)

1. The student generates on-policy responses to shared prompts.
2. Both teacher and student run forward passes; hidden states are collected and saved as `on_policy_pairs.jsonl`.

**Stage 1: Bridge Construction** (`scripts/analysis/run_cross_arch_preexp2.sh`)

1. Load on-policy pairs from Stage 0.
2. Compute teacher PCA bases $P_T$ (top-$r$ principal directions of teacher hidden-state covariance).
3. Train student projectors $P_S$ to minimize reconstruction error in the shared $r$-dimensional subspace.
4. Freeze both $P_T$ and $P_S$ as the bridge.

**Stage 2: Distillation** (`low_rank_rep_distillation.sh`)

- The frozen bridge projects both teacher and student hidden states into a shared low-rank subspace.
- The student LLM is trained with MSE loss in this aligned subspace.
- The bridge rank $r$ controls the bandwidth of the supervision channel (default: $r=8$ for cross-architecture, $r=4$ for cross-tokenizer).

Key parameters:

| Parameter                      | Default | Description                                      |
| ------------------------------ | ------- | ------------------------------------------------ |
| `REP_PROJECTOR_MODE`           | `low_rank` | Bridge mode (`full` for vanilla, `low_rank` for bridge) |
| `REP_LOW_RANK`                 | `4`     | Bridge rank $r$                                  |
| `REP_LOW_RANK_INIT_CHECKPOINT` | —       | Path to pre-trained bridge (`ps_bank.pt`)        |
| `REP_FREEZE_PS`                | `True`  | Freeze bridge during Stage 2                     |

### Token-level OPD (baseline)

The original **top-*k*** / sampled-token OPD path remains available via `on_policy_distillation.sh` (`LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, etc.) for comparisons.

---

## Getting Started

### Environment setup

Training is based on [verl](https://github.com/verl-project/verl) (v0.7.0), inherited from the OPD release:

```bash
conda create -n verl python==3.12
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

### Model paths

Set the following environment variables before running any script:

```bash
export MODEL_DIR=/path/to/your/models      # e.g., HuggingFace model directories
export DATA_DIR=/path/to/your/datasets      # training/eval data
```

### Training

#### OPRD-Vanilla (same architecture, recommended)

```bash
# Rep-only (no OPD loss)
bash rep_distillation.sh

# Combined: OPD + representation
REP_DISTILLATION_ONLY=False LOG_PROB_TOP_K=16 bash rep_distillation.sh
```

#### OPRD-Bridge (cross-architecture / cross-tokenizer)

```bash
# Stage 1: Generate on-policy pairs (student rollouts + teacher hidden states)
bash scripts/analysis/run_cross_arch_analysis.sh

# Stage 2: Build the bridge (train P_S with frozen teacher PCA bases)
bash scripts/analysis/run_cross_arch_preexp2.sh

# Stage 3: Distill with frozen bridge
bash low_rank_rep_distillation.sh
```

#### Token-level OPD only (upstream-style)

```bash
USE_REP_DISTILLATION=False bash on_policy_distillation.sh
```

#### GRPO (RL baseline, no distillation)

```bash
bash grpo.sh
```

> [!NOTE]
> For non-thinking models (e.g. Qwen3-1.7B non-thinking), add `+data.apply_chat_template_kwargs.enable_thinking=False` to the training command.

---

## Memory profiling

To compare **rep-only** vs **top-*k* OPD** actor-update peak memory:

```bash
export ACTOR_UPDATE_MEM_PROFILE=1
bash rep_distillation.sh   # or on_policy_distillation.sh with desired LOG_PROB_TOP_K
```

Logged metrics (per step, `all_reduce(MAX)` over ranks):

- `mem/actor_update_peak_alloc_GB` — peak allocated during `update_policy`
- `mem/actor_update_delta_peak_GB` — transient update pressure (peak minus baseline)
- `mem/actor_update_peak_reserved_GB` — allocator reserved peak

---

## Validation

Evaluation follows the [JustRL](https://github.com/thunlp/JustRL) pipeline:

```bash
cd scripts/val/eval
python gen_vllm.py   # set MODEL_NAMES and workers
python grade.py
```

---

## Repository layout

| Path                                  | Role                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| `rep_distillation.sh`                 | **OPRD-Vanilla** launcher (same-architecture)                |
| `low_rank_rep_distillation.sh`        | **OPRD-Bridge** launcher (cross-architecture/cross-tokenizer)|
| `on_policy_distillation.sh`           | Shared verl training driver (OPD + OPRD)                     |
| `grpo.sh`                             | GRPO baseline (no distillation)                              |
| `scripts/analysis/`                   | Bridge construction scripts (Stage 1)                        |
| `verl/verl/utils/rep_distillation.py` | Representation extract / loss / layer alignment              |
| `verl/verl/workers/actor/dp_actor.py` | Actor update: PG, rep losses                                 |
| `verl/verl/workers/fsdp_workers.py`   | Teacher hidden cache, memory profiling                       |

---

## Citation

If you use **OPRD** from this repository, please cite:

```bibtex
@article{yang2026oprd,
  title={OPRD: On-Policy Representation Distillation},
  author={Yang, Shenzhi and Zhu, Guangcheng and Song, Bowen and Wang, Haobo and Xia, Mingxuan and Zheng, Xing and Ma, Yingfan and Chen, Zhongqi and Wang, Weiqiang and Chen, Gang},
  journal={arXiv preprint arXiv:2606.06021},
  year={2026}
}

@article{li2026rethinking,
  title={Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe},
  author={Li, Yaxuan and Zuo, Yuxin and He, Bingxiang and Zhang, Jinqian and Xiao, Chaojun and Qian, Cheng and Yu, Tianyu and Gao, Huan-ang and Yang, Wenkai and Liu, Zhiyuan and Ding, Ning},
  journal={arXiv preprint arXiv:2604.13016},
  year={2026}
}
```

---

## Acknowledgments

This repository extends the open-source implementation of **On-Policy Distillation (OPD)** from:

> **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**  
> [Paper](https://arxiv.org/abs/2604.13016) | [GitHub (thunlp/OPD)](https://github.com/thunlp/OPD)

We thank the OPD authors for the training recipe, analysis, and verl-based codebase that this project builds upon.
