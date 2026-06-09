# OPRD: On-Policy Representation Distillation

**Representation-level on-policy distillation for large language models, built on the [OPD](https://github.com/thunlp/OPD) training stack.**

[Paper (OPRD)](https://arxiv.org/abs/2606.06021)
[Upstream OPD](https://github.com/thunlp/OPD)
[Paper (OPD)](https://arxiv.org/abs/2604.13016)
[verl](https://github.com/verl-project/verl)

**[📖 Overview](#overview)** • **[🔬 Method](#method)** • **[✨ Getting Started](#getting-started)**

**[📊 Memory](#memory-profiling)** • **[✅ Validation](#validation)** • **[🎈 Citation](#citation)** • **[🙏 Acknowledgments](#acknowledgments)**

---

## 📖 Overview

**OPRD** distills a **teacher** into a **student** during on-policy rollouts by matching **hidden representations** on student-generated responses, instead of (or in addition to) matching token-level log-probabilities over a large vocabulary.

Compared to top-*k* token OPD on long chain-of-thought responses, OPRD typically:

- Avoids materializing full-vocabulary `log_softmax` during the actor update
- Supports **multi-layer** alignment (with proportional layer mapping when student/teacher depths differ)
- Uses compact position modes (`last_k`, `first_k`, etc.) for memory-efficient batching

This repo also contains an **attention distillation** prototype (`USE_ATT_DISTILLATION`); it is **experimental and not yet production-ready**—use at your own risk.

---

## 🔬 Method

### On-Policy Representation Distillation (OPRD)

1. **Rollout**: The student generates on-policy responses (same pipeline as OPD).
2. **Teacher cache**: The teacher runs a forward pass; per-layer hidden states on the response region are stored (e.g. `teacher_last_hidden_repr`).
3. **Student update**: The student forward produces matching hidden states; the loss is **MSE** (and logged **cosine similarity**) between student and teacher representations, optionally projected when hidden sizes differ.

Key knobs (see `rep_distillation.sh`):


| Concept             | Env vars                                  | Options                                             |
| ------------------- | ----------------------------------------- | --------------------------------------------------- |
| Token positions     | `REP_DISTILLATION_POSITIONS`              | `last`, `all`, `last_k`, `first_k`                  |
| Layers              | `REP_DISTILLATION_LAYERS`                 | `last`, `all`, `even`, `odd`                        |
| Rep-only vs OPD+rep | `REP_DISTILLATION_ONLY`, `LOG_PROB_TOP_K` | `True` + `0` = rep-only; `False` + `K>0` = combined |


### Token-level OPD (baseline)

The original **top-*k*** / sampled-token OPD path remains available via `on_policy_distillation.sh` (`LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, etc.) for apples-to-apples comparisons.

### Attention distillation (WIP)

Attention-row distillation between student and teacher is implemented in development (`USE_ATT_DISTILLATION`, `att_distillation.sh` patterns in `on_policy_distillation.sh`). Known constraints include eager attention, context slicing for padded prompts, and higher backward memory. **Not recommended for main experiments until documented and stabilized.**

---

## ✨ Getting Started

### Environment setup

Training is based on [verl](https://github.com/verl-project/verl) (v0.7.0), inherited from the OPD release:

```bash
conda create -n verl python==3.12
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

Optional SFT (LlamaFactory, as in upstream OPD):

```bash
conda create -n sft python==3.11
cd LlamaFactory/
pip install -e .
pip install -r requirements/metrics.txt
```

### Training

#### OPRD (recommended entrypoint)

```bash
bash rep_distillation.sh
```

Defaults in `rep_distillation.sh`: representation distillation on, **rep-only** (`REP_DISTILLATION_ONLY=True`, `LOG_PROB_TOP_K=0`). Edit exports there or override env vars before launch.

#### OPD + representation (combined)

```bash
REP_DISTILLATION_ONLY=False LOG_PROB_TOP_K=16 bash rep_distillation.sh
```

#### Token-level OPD only (upstream-style)

```bash
USE_REP_DISTILLATION=False bash on_policy_distillation.sh
```

**OPRD / representation parameters**


| Parameter                             | Typical default                    | Description                                                         |
| ------------------------------------- | ---------------------------------- | ------------------------------------------------------------------- |
| `USE_REP_DISTILLATION`                | `True` (via `rep_distillation.sh`) | Enable representation distillation                                  |
| `REP_DISTILLATION_ONLY`               | `True`                             | Skip policy-gradient / top-*k* loss; optimize rep loss only         |
| `REP_DISTILLATION_COEF`               | `1.0`                              | Loss coefficient                                                    |
| `REP_DISTILLATION_POSITIONS`          | `last_k`                           | Where on the response to match (`last`, `all`, `last_k`, `first_k`) |
| `REP_DISTILLATION_LAST_K` / `FIRST_K` | `2000`                             | Width for compact `last_k` / `first_k` modes                        |
| `REP_DISTILLATION_LAYERS`             | `all`                              | Which transformer layers to distill                                 |
| `ACTOR_MODEL_PATH`                    | —                                  | Student (policy) model                                              |
| `REWARD_MODEL_PATH`                   | —                                  | Teacher model (hidden states + rewards)                             |


Cross-architecture (e.g. 1.7B student, 4B teacher): hidden dims are aligned via a learned **rep projector**; layer counts are aligned via **proportional layer indexing** when depths differ.

**Token-level OPD parameters (baseline)**


| Parameter                               | Default               | Description                                                           |
| --------------------------------------- | --------------------- | --------------------------------------------------------------------- |
| `ADV_ESTIMATOR`                         | `token_reward_direct` | Required for OPD-style training                                       |
| `LOG_PROB_TOP_K`                        | `16`                  | Top-*k* token rewards; `0` = sampled-token OPD                        |
| `TOP_K_STRATEGY`                        | `only_stu`            | `only_stu`, `only_tch`, `intersection`, `union`, `union-intersection` |
| `REWARD_WEIGHT_MODE`                    | `student_p`           | `student_p`, `teacher_p`, `none`                                      |
| `N_RESPONSES`                           | `2`                   | Rollouts per prompt                                                   |
| `MAX_PROMPT_LENGTH` / `MAX_RESP_LENGTH` | —                     | Sequence limits (long CoT configs use larger values in script)        |


**Attention distillation (experimental)**


| Parameter                                           | Description                             |
| --------------------------------------------------- | --------------------------------------- |
| `USE_ATT_DISTILLATION`                              | Enable attention-row distillation (WIP) |
| `ATT_DISTILLATION_COEF`                             | Loss weight                             |
| `ATT_DISTILLATION_POSITIONS` / `LAST_K` / `FIRST_K` | Same spirit as rep positions            |
| `ATT_DISTILLATION_LAYERS`                           | `last` or `all`                         |
| `ATT_DISTILLATION_MAX_KEY_LEN`                      | Cap causal key length                   |
| `ATT_DISTILLATION_LOSS`                             | `kl` or `mse`                           |


> [!NOTE]
> For non-thinking models (e.g. Qwen3-1.7B non-thinking), add `+data.apply_chat_template_kwargs.enable_thinking=False` to the training command, as in the upstream OPD README.

> [!TIP]
> Deduplicate DeepMath vs DAPO-Math-17K with `scripts/infer/dedup_deepmath.py` when reproducing OPD paper data settings.

### SFT & RL

Teacher rollout for offline SFT and GRPO recipes are unchanged from the OPD release; see upstream [OPD README](https://github.com/thunlp/OPD) and `grpo.sh` (`ADV_ESTIMATOR=grpo`, `LOG_PROB_TOP_K=0`).

---

## 📊 Memory profiling

To compare **rep-only** vs **top-*k* OPD** actor-update peak memory on the worst GPU:

```bash
export ACTOR_UPDATE_MEM_PROFILE=1
bash rep_distillation.sh   # or on_policy_distillation.sh with desired LOG_PROB_TOP_K
```

Logged metrics (per step, `all_reduce(MAX)` over ranks):

- `mem/actor_update_peak_alloc_GB` — peak allocated during `update_policy`
- `mem/actor_update_delta_peak_GB` — peak minus memory at segment start (transient update pressure)
- `mem/actor_update_peak_reserved_GB` — allocator reserved peak

`delta_peak` is **per-GPU**, reported as the **maximum across ranks**, not the sum of all cards.

---

## ✅ Validation

Evaluation follows the [JustRL](https://github.com/thunlp/JustRL) pipeline (from OPD):

```bash
cd scripts/val/eval
python gen_vllm.py   # set MODEL_NAMES and workers
python grade.py
# optional: python grade.py --enable_model_verifier
```

*Development experiments use multi-GPU nodes (e.g. 8× NVIDIA A100 80GB); adjust batch and `last_k` for your hardware.*

---

## 🎈 Citation

If you use **OPRD** from this repository, please cite this work and the OPD paper:

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

## Repository layout (high level)


| Path                                  | Role                                                    |
| ------------------------------------- | ------------------------------------------------------- |
| `rep_distillation.sh`                 | **OPRD** launcher (rep defaults)                        |
| `on_policy_distillation.sh`           | Shared verl training driver (OPD + OPRD + optional att) |
| `verl/verl/utils/rep_distillation.py` | Representation extract / loss / layer alignment         |
| `verl/verl/utils/att_distillation.py` | Attention distillation (**WIP**)                        |
| `verl/verl/workers/actor/dp_actor.py` | Actor update: PG, rep, att losses                       |
| `verl/verl/workers/fsdp_workers.py`   | Teacher hidden/attn cache, actor memory profiling       |


---

## 🙏 Acknowledgments

This repository extends the open-source implementation of **On-Policy Distillation (OPD)** from:

> **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**  
> [Paper](https://arxiv.org/abs/2604.13016) · [GitHub (thunlp/OPD)](https://github.com/thunlp/OPD)

We thank the OPD authors for the training recipe, analysis, and verl-based codebase that this project builds upon. **OPRD** (On-Policy Representation Distillation) is our method; token-level OPD and auxiliary tooling in this repo follow their design unless noted otherwise.