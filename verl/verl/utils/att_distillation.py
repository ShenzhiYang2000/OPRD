# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention distillation utilities for on-policy distillation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from verl.utils.rep_distillation import (
    align_teacher_layers_to_student,
    build_compact_rep_distillation_position_mask,
    build_rep_distillation_position_mask,
    get_batch_distillation_k,
    get_compact_distillation_width,
    get_per_sample_distillation_k,
    get_response_valid_token_counts,
    validate_rep_distillation_layers,
    validate_rep_distillation_positions,
)

AttDistillationLossType = Literal["kl", "mse"]
VALID_ATT_DISTILLATION_LOSS_TYPES = ("kl", "mse")


def validate_att_distillation_loss_type(loss_type: str) -> str:
    if loss_type not in VALID_ATT_DISTILLATION_LOSS_TYPES:
        raise ValueError(
            f"att_distillation_loss must be one of {VALID_ATT_DISTILLATION_LOSS_TYPES}, got {loss_type!r}"
        )
    return loss_type


def _resolve_hf_causal_lm(model: nn.Module) -> nn.Module:
    """Unwrap FSDP / DDP shells and return the HF causal LM root."""
    current: nn.Module | None = model
    seen: set[int] = set()
    for _ in range(12):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, "config") and hasattr(current, "model"):
            return current
        if hasattr(current, "_fsdp_wrapped_module"):
            current = current._fsdp_wrapped_module
        elif hasattr(current, "module"):
            current = current.module
        else:
            break
    fallback = getattr(model, "module", model)
    if hasattr(fallback, "config"):
        return fallback
    raise ValueError("Could not resolve HF causal LM module for attention distillation")


def _iter_attn_configs(model: nn.Module):
    root = _resolve_hf_causal_lm(model)
    configs = [root.config]
    text_config = getattr(root.config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    vision_config = getattr(root.config, "vision_config", None)
    if vision_config is not None:
        configs.append(vision_config)
    return configs


@contextmanager
def disable_gradient_checkpointing(model: nn.Module):
    """Disable HF gradient checkpointing for a short eager-attention forward."""
    hf_model = _resolve_hf_causal_lm(model)
    grad_ckpt_enabled = getattr(hf_model, "is_gradient_checkpointing", False)
    if grad_ckpt_enabled:
        hf_model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if grad_ckpt_enabled:
            hf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


@contextmanager
def eager_attention_context(model: nn.Module):
    """Temporarily switch HF model attention to eager so ``output_attentions=True`` works."""
    old_values: dict[tuple[int, str], object] = {}
    configs = _iter_attn_configs(model)
    for config in configs:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(config, attr):
                key = (id(config), attr)
                old_values[key] = getattr(config, attr)
                setattr(config, attr, "eager")
    try:
        yield
    finally:
        for config in configs:
            for attr in ("_attn_implementation", "attn_implementation"):
                key = (id(config), attr)
                if key in old_values:
                    setattr(config, attr, old_values[key])


def _get_max_valid_prompt_len(attention_mask: torch.Tensor, response_len: int) -> int:
    """Return the longest valid (non-padding) prompt length in the batch."""
    if response_len <= 0 or attention_mask.size(1) <= response_len:
        return int(attention_mask.sum(dim=1).max().item())
    prompt_mask = attention_mask[:, :-response_len]
    if prompt_mask.numel() == 0:
        return 0
    return int(prompt_mask.sum(dim=1).max().item())


def slice_inputs_for_att_distillation(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_len: int,
    max_context_len: int,
    positions: str = "last",
    first_k: int = 50,
    last_k: int = 32,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Slice inputs for memory-efficient attention extraction."""
    validate_rep_distillation_positions(positions)
    seqlen = input_ids.size(1)
    response_start = seqlen - response_len
    max_prompt_len = _get_max_valid_prompt_len(attention_mask, response_len)

    if positions == "first_k":
        if response_mask is not None:
            effective_k = get_batch_distillation_k(response_mask, first_k)
        else:
            effective_k = min(int(first_k), response_len)
        end = min(seqlen, response_start + effective_k)
        start = max(0, end - max_context_len)
    elif positions == "last_k":
        if response_mask is not None:
            max_valid = int(get_response_valid_token_counts(response_mask).max().item())
            eff_k = (
                min(get_compact_distillation_width(last_k), max_valid)
                if max_valid > 0
                else 0
            )
            end = min(seqlen, response_start + max_valid)
        else:
            eff_k = min(int(last_k), response_len)
            end = seqlen
        # Use valid prompt length, not padded response_start (left-padded prompts).
        desired_len = min(max_context_len, max_prompt_len + eff_k)
        start = max(0, end - desired_len)
    else:
        context_len = min(seqlen, max_context_len)
        start = seqlen - context_len
        end = seqlen

    return (
        input_ids[:, start:end],
        attention_mask[:, start:end],
        position_ids[..., start:end] if position_ids.dim() >= 2 else position_ids[:, start:end],
        start,
    )


def get_att_distillation_context_metadata(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    positions: str,
    first_k: int = 50,
    last_k: int = 32,
    max_context_len: int = 4096,
) -> tuple[int, int, int]:
    """Return ``(original_seqlen, context_start, context_len)`` for att row extraction."""
    response_len = response_mask.size(1)
    original_seqlen = input_ids.size(1)
    sliced_ids, _, _, context_start = slice_inputs_for_att_distillation(
        input_ids,
        attention_mask,
        position_ids,
        response_len,
        max_context_len,
        positions=positions,
        first_k=first_k,
        last_k=last_k,
        response_mask=response_mask,
    )
    return original_seqlen, context_start, sliced_ids.size(1)


def get_att_distillation_batch_key_width(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    positions: str,
    first_k: int = 50,
    last_k: int = 32,
    max_context_len: int = 4096,
) -> int:
    """Return the batched key width for stacked att rows on this rank."""
    _, _, context_len = get_att_distillation_context_metadata(
        input_ids,
        attention_mask,
        position_ids,
        response_mask,
        positions=positions,
        first_k=first_k,
        last_k=last_k,
        max_context_len=max_context_len,
    )
    return context_len


def sync_att_distillation_batch_key_width(local_key_width: int, device: torch.device) -> int:
    """All-reduce MAX key width so DP ranks can ``DataProto.concat`` att rows."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_key_width
    width = torch.tensor([local_key_width], device=device, dtype=torch.long)
    torch.distributed.all_reduce(width, op=torch.distributed.ReduceOp.MAX)
    return int(width.item())


def pad_attn_rows_to_key_width(rows: torch.Tensor, key_width: int) -> torch.Tensor:
    """Right-align causal att rows by left-padding the key dimension."""
    if key_width <= 0:
        raise ValueError(f"key_width must be positive, got {key_width}")
    current_width = rows.size(-1)
    if current_width == key_width:
        return rows
    if current_width > key_width:
        return rows[..., -key_width:]
    pad_left = key_width - current_width
    return F.pad(rows, (pad_left, 0))


def _get_response_query_token_positions(
    response_mask: torch.Tensor,
    positions: str,
    *,
    last_k: int = 32,
    first_k: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-batch response token indices and mask for att query extraction."""
    batch_size, response_len = response_mask.shape
    device = response_mask.device

    if positions == "first_k":
        k = get_compact_distillation_width(first_k)
        if k <= 0:
            return (
                torch.zeros(batch_size, 0, device=device, dtype=torch.long),
                response_mask.new_zeros(batch_size, 0),
            )
        token_pos = torch.arange(k, device=device).unsqueeze(0).expand(batch_size, k)
        position_mask = build_compact_rep_distillation_position_mask(
            response_mask, positions, last_k=last_k, first_k=first_k
        )
        return token_pos, position_mask

    if positions == "last_k":
        k = get_compact_distillation_width(last_k)
        if k <= 0:
            return (
                torch.zeros(batch_size, 0, device=device, dtype=torch.long),
                response_mask.new_zeros(batch_size, 0),
            )
        per_sample_k = get_per_sample_distillation_k(response_mask, last_k)
        token_pos = torch.arange(k, device=device).unsqueeze(0).expand(batch_size, k)
        last_valid = get_response_valid_token_counts(response_mask)
        lower = last_valid - per_sample_k
        token_pos = lower.unsqueeze(1) + token_pos
        position_mask = build_compact_rep_distillation_position_mask(
            response_mask, positions, last_k=last_k, first_k=first_k
        )
        return token_pos, position_mask

    token_pos = torch.arange(response_len, device=device).unsqueeze(0).expand(batch_size, response_len)
    position_mask = build_rep_distillation_position_mask(
        response_mask, positions, last_k=last_k, first_k=first_k
    )
    return token_pos, position_mask


def _get_absolute_response_query_indices(
    response_mask: torch.Tensor,
    positions: str,
    last_k: int,
    context_start: int,
    context_len: int,
    first_k: int = 50,
    *,
    original_seqlen: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map response queries to indices inside the truncated context tensor."""
    validate_rep_distillation_positions(positions)
    batch_size, response_len = response_mask.shape
    # ``context_start + context_len`` is the slice end in original coordinates (not full
    # sequence length). For ``first_k`` we truncate before the response tail, so response
    # start must come from the original sequence length.
    if original_seqlen is None:
        original_seqlen = context_start + context_len
    response_start = original_seqlen - response_len

    if positions == "last":
        last_in_response = response_mask.long().sum(dim=1) - 1
        last_in_response = last_in_response.clamp(min=0)
        abs_idx = response_start + last_in_response
        ctx_idx = abs_idx - context_start
        return ctx_idx, None

    token_pos, position_mask = _get_response_query_token_positions(
        response_mask, positions, last_k=last_k, first_k=first_k
    )
    abs_idx = response_start + token_pos
    ctx_idx = abs_idx - context_start
    valid = (ctx_idx >= 0) & (ctx_idx < context_len) & position_mask.bool()
    ctx_idx = ctx_idx.clamp(min=0, max=context_len - 1)
    return ctx_idx, valid.float()


def att_distillation_query_valid_fraction(
    response_mask: torch.Tensor,
    positions: str,
    context_start: int,
    context_len: int,
    *,
    last_k: int = 32,
    first_k: int = 50,
    original_seqlen: int | None = None,
) -> float:
    """Fraction of selected query positions that fall inside the truncated context."""
    _, query_valid_mask = _get_absolute_response_query_indices(
        response_mask,
        positions,
        last_k,
        context_start,
        context_len,
        first_k=first_k,
        original_seqlen=original_seqlen,
    )
    if query_valid_mask is None:
        return 1.0
    _, position_mask = _get_response_query_token_positions(
        response_mask, positions, last_k=last_k, first_k=first_k
    )
    denom = position_mask.sum().clamp(min=1.0)
    return float((query_valid_mask * position_mask).sum().item() / denom.item())


def _extract_rows_from_layer_attn(
    attn: torch.Tensor,
    query_ctx_idx: torch.Tensor,
    query_valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Extract causal attention rows for selected queries.

    Args:
        attn: (B, H, S, S)
        query_ctx_idx: (B,) or (B, T)
        query_valid_mask: optional (B, T)

    Returns:
        (B, H, K) for single-query, or (B, T, H, K) for multi-query.
    """
    batch_size, num_heads, seqlen, _ = attn.shape
    if query_ctx_idx.shape[0] != batch_size:
        raise ValueError(
            f"Attention batch size {batch_size} != query index batch size {query_ctx_idx.shape[0]}"
        )
    if query_ctx_idx.dim() == 1:
        rows = []
        for batch_idx in range(batch_size):
            q_idx = int(query_ctx_idx[batch_idx].item())
            row = attn[batch_idx, :, q_idx, : q_idx + 1]
            rows.append(row)
        max_key_len = max(int(query_ctx_idx[batch_idx].item()) + 1 for batch_idx in range(batch_size))
        padded = attn.new_zeros(batch_size, num_heads, max_key_len)
        for batch_idx, row in enumerate(rows):
            padded[batch_idx, :, : row.size(-1)] = row
        return padded

    num_queries = query_ctx_idx.size(1)
    max_key_len = 0
    row_list: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        batch_rows = []
        for query_idx in range(num_queries):
            if query_valid_mask is not None and query_valid_mask[batch_idx, query_idx] <= 0:
                batch_rows.append(attn.new_zeros(num_heads, 1))
                continue
            q_idx = int(query_ctx_idx[batch_idx, query_idx].item())
            batch_rows.append(attn[batch_idx, :, q_idx, : q_idx + 1])
            max_key_len = max(max_key_len, batch_rows[-1].size(-1))
        row_list.append(batch_rows)

    padded = attn.new_zeros(batch_size, num_queries, num_heads, max(max_key_len, 1))
    for batch_idx, batch_rows in enumerate(row_list):
        for query_idx, row in enumerate(batch_rows):
            padded[batch_idx, query_idx, :, : row.size(-1)] = row
    return padded


def get_att_distillation_layer_indices(num_layers: int, layers: str) -> list[int]:
    """Map layer mode to indices in HF ``attentions`` tuple (one entry per block)."""
    validate_rep_distillation_layers(layers)
    if num_layers < 1:
        raise ValueError(f"Expected at least one attention layer, got {num_layers}")
    if layers == "last":
        return [num_layers - 1]
    if layers == "all":
        return list(range(num_layers))
    if layers == "even":
        return list(range(0, num_layers, 2))
    return list(range(1, num_layers, 2))


def extract_teacher_response_attn_rows(
    attentions: tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
    positions: str,
    layers: str,
    *,
    context_start: int,
    context_len: int,
    last_k: int = 32,
    first_k: int = 50,
    original_seqlen: int | None = None,
) -> torch.Tensor:
    """Extract stacked teacher attention rows from HF ``attentions`` tuple."""
    validate_rep_distillation_layers(layers)
    layer_indices = get_att_distillation_layer_indices(len(attentions), layers)

    query_ctx_idx, query_valid_mask = _get_absolute_response_query_indices(
        response_mask,
        positions,
        last_k,
        context_start,
        context_len,
        first_k=first_k,
        original_seqlen=original_seqlen,
    )
    layer_rows = [
        _extract_rows_from_layer_attn(attentions[idx].float(), query_ctx_idx, query_valid_mask)
        for idx in layer_indices
    ]
    if len(layer_rows) == 1:
        return layer_rows[0]
    return torch.stack(layer_rows, dim=1)


def _align_student_teacher_attn_rows(
    student_rows: torch.Tensor,
    teacher_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-align key dimension so different causal row lengths match."""
    if student_rows.shape == teacher_rows.shape:
        return student_rows, teacher_rows
    key_dim = max(student_rows.size(-1), teacher_rows.size(-1))
    student_pad = key_dim - student_rows.size(-1)
    teacher_pad = key_dim - teacher_rows.size(-1)
    if student_pad > 0:
        student_rows = F.pad(student_rows, (student_pad, 0))
    if teacher_pad > 0:
        teacher_rows = F.pad(teacher_rows, (teacher_pad, 0))
    return student_rows, teacher_rows


def attention_distillation_loss(
    student_rows: torch.Tensor,
    teacher_rows: torch.Tensor,
    *,
    loss_type: str = "kl",
    position_mask: torch.Tensor | None = None,
    loss_agg_mode: str = "token-mean",
    temperature: float = 1.0,
    num_layers: int | None = None,
) -> torch.Tensor:
    """Distill causal attention rows with KL or MSE over the key dimension."""
    validate_att_distillation_loss_type(loss_type)
    student_rows, teacher_rows = _align_student_teacher_attn_rows(student_rows, teacher_rows)
    teacher_rows = teacher_rows.detach()

    if num_layers is not None and num_layers > 1 and student_rows.dim() >= 4 and teacher_rows.dim() >= 4:
        student_rows, teacher_rows = align_teacher_layers_to_student(student_rows, teacher_rows)
        num_layers = student_rows.size(1)

    if num_layers is not None and num_layers > 1 and student_rows.dim() >= 4 and student_rows.size(1) == num_layers:
        layer_losses = [
            attention_distillation_loss(
                student_rows[:, layer_idx],
                teacher_rows[:, layer_idx],
                loss_type=loss_type,
                position_mask=position_mask,
                loss_agg_mode=loss_agg_mode,
                temperature=temperature,
            )
            for layer_idx in range(num_layers)
        ]
        return torch.stack(layer_losses).mean()

    if loss_type == "mse":
        per_query = ((student_rows - teacher_rows) ** 2).mean(dim=(-2, -1))
    else:
        student_log_probs = F.log_softmax(student_rows / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_rows / temperature, dim=-1)
        per_query = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean(dim=-1)

    if per_query.dim() == 1:
        return per_query.mean()

    if position_mask is None:
        position_mask = torch.ones_like(per_query)
    from verl.trainer.ppo.core_algos import agg_loss

    return agg_loss(per_query, position_mask, loss_agg_mode)


def forward_response_attn_rows(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    multi_modal_inputs: dict | None = None,
    positions: str = "last",
    layers: str = "last",
    last_k: int = 32,
    first_k: int = 50,
    max_context_len: int = 4096,
) -> torch.Tensor:
    """Run eager attention forward on a truncated context and return response query rows."""
    multi_modal_inputs = multi_modal_inputs or {}
    response_len = response_mask.size(1)
    original_seqlen = input_ids.size(1)
    input_ids, attention_mask, position_ids, context_start = slice_inputs_for_att_distillation(
        input_ids,
        attention_mask,
        position_ids,
        response_len,
        max_context_len,
        positions=positions,
        first_k=first_k,
        last_k=last_k,
        response_mask=response_mask,
    )

    with disable_gradient_checkpointing(model):
        with eager_attention_context(model):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=True,
                **multi_modal_inputs,
            )

    if output.attentions is None:
        raise RuntimeError(
            "Attention distillation requires model attentions; ensure attn_implementation supports output_attentions"
        )

    rows = extract_teacher_response_attn_rows(
        output.attentions,
        response_mask,
        positions,
        layers,
        context_start=context_start,
        context_len=input_ids.size(1),
        last_k=last_k,
        first_k=first_k,
        original_seqlen=original_seqlen,
    )
    del output
    return rows
