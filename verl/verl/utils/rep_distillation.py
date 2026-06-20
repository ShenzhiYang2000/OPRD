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

"""Representation-level on-policy distillation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_name
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs

RepDistillationPositions = Literal["last", "all", "last_k", "first_k"]
VALID_REP_DISTILLATION_POSITIONS = ("last", "all", "last_k", "first_k")

RepDistillationLayers = Literal["last", "all", "even", "odd"]
VALID_REP_DISTILLATION_LAYERS = ("last", "all", "even", "odd")

RepProjectorMode = Literal["full", "low_rank", "low_rank_residual"]
VALID_REP_PROJECTOR_MODES = ("full", "low_rank", "low_rank_residual")


def validate_rep_distillation_layers(layers: str) -> str:
    if layers not in VALID_REP_DISTILLATION_LAYERS:
        raise ValueError(
            f"rep_distillation_layers must be one of {VALID_REP_DISTILLATION_LAYERS}, got {layers!r}"
        )
    return layers


def get_rep_distillation_hidden_state_indices(num_hidden_states: int, layers: str) -> list[int]:
    """Map layer mode to indices in HF ``hidden_states`` (0=embeddings, 1..L=block outputs)."""
    validate_rep_distillation_layers(layers)
    if num_hidden_states < 2:
        raise ValueError(f"Expected at least embedding + one block in hidden_states, got {num_hidden_states}")

    if layers == "last":
        return [num_hidden_states - 1]

    num_transformer_layers = num_hidden_states - 1
    if layers == "all":
        return list(range(1, num_hidden_states))
    if layers == "even":
        return [1 + layer_idx for layer_idx in range(0, num_transformer_layers, 2)]
    # odd
    return [1 + layer_idx for layer_idx in range(1, num_transformer_layers, 2)]


def validate_rep_distillation_positions(positions: str) -> str:
    if positions not in VALID_REP_DISTILLATION_POSITIONS:
        raise ValueError(
            f"rep_distillation_positions must be one of {VALID_REP_DISTILLATION_POSITIONS}, got {positions!r}"
        )
    return positions


def get_response_valid_token_counts(response_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample number of valid response tokens, shape ``(batch,)``."""
    return response_mask.long().sum(dim=1)


def get_batch_distillation_k(response_mask: torch.Tensor, k: int) -> int:
    """Context slice width for att: ``min(k, max valid response len in batch)``."""
    k = int(k)
    if k <= 0:
        return 0
    max_valid = int(get_response_valid_token_counts(response_mask).max().item())
    if max_valid <= 0:
        return 0
    return min(k, max_valid)


def get_compact_distillation_width(k: int) -> int:
    """Fixed compact tensor width for ``first_k`` / ``last_k`` storage across micro-batches."""
    return max(int(k), 0)


def get_per_sample_distillation_k(response_mask: torch.Tensor, k: int) -> torch.Tensor:
    """Per-sample effective k: ``min(k, valid_len)`` for each row in ``response_mask``."""
    k = int(k)
    valid_counts = get_response_valid_token_counts(response_mask)
    if k <= 0:
        return torch.zeros_like(valid_counts)
    return torch.minimum(valid_counts, torch.full_like(valid_counts, k))


def extract_response_hidden_states(
    hidden_states: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract last-layer hidden states over the full response segment.

    Args:
        hidden_states: (batch, seqlen, hidden_dim)
        response_mask: (batch, response_len)

    Returns:
        (batch, response_len, hidden_dim)
    """
    response_len = response_mask.size(1)
    return hidden_states[:, -response_len:, :]


def extract_response_last_valid_token_hidden_states(
    hidden_states: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract last-layer hidden states at the last valid token in the response segment.

    Assumes ``responses`` are right-aligned at the end of the full sequence, which matches
    verl's ``[left-padded prompt | response]`` layout.

    Args:
        hidden_states: (batch, seqlen, hidden_dim)
        response_mask: (batch, response_len), 1 for valid generated tokens

    Returns:
        (batch, hidden_dim)
    """
    response_hidden = extract_response_hidden_states(hidden_states, response_mask)
    last_indices = response_mask.long().sum(dim=1) - 1
    last_indices = last_indices.clamp(min=0)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return response_hidden[batch_indices, last_indices]


def build_rep_distillation_position_mask(
    response_mask: torch.Tensor,
    positions: str,
    last_k: int = 32,
    first_k: int = 50,
) -> torch.Tensor:
    """Build a (batch, response_len) mask selecting tokens for rep distillation."""
    validate_rep_distillation_positions(positions)
    response_mask = response_mask.float()
    if positions == "all":
        return response_mask

    response_len = response_mask.size(1)
    batch_size = response_mask.size(0)
    device = response_mask.device
    token_pos = torch.arange(response_len, device=device).unsqueeze(0)

    if positions == "last":
        mask = torch.zeros(batch_size, response_len, device=device, dtype=response_mask.dtype)
        last_indices = response_mask.long().sum(dim=1) - 1
        last_indices = last_indices.clamp(min=0)
        batch_indices = torch.arange(batch_size, device=device)
        mask[batch_indices, last_indices] = 1.0
        return mask * response_mask

    last_valid = response_mask.long().sum(dim=1, keepdim=True)

    if positions == "first_k":
        effective_k = get_per_sample_distillation_k(response_mask, first_k).unsqueeze(1)
        in_first_k = token_pos < effective_k
        return response_mask * in_first_k.float()

    # last_k: per sample use min(last_k, valid_len) trailing tokens
    effective_k = get_per_sample_distillation_k(response_mask, last_k).unsqueeze(1)
    lower = (last_valid - effective_k).clamp(min=0)
    in_last_k = (token_pos >= lower) & (token_pos < last_valid)
    return response_mask * in_last_k.float()


def _compact_single_layer_response_repr(
    repr: torch.Tensor,
    response_mask: torch.Tensor,
    position_mask: torch.Tensor,
    positions: str,
    last_k: int,
    first_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact ``(B, T, D)`` to selected response tokens only."""
    batch_size = repr.size(0)
    last_valid = get_response_valid_token_counts(response_mask)

    if positions == "first_k":
        k_batch = get_compact_distillation_width(first_k)
        per_sample_k = get_per_sample_distillation_k(response_mask, first_k)
    else:
        k_batch = get_compact_distillation_width(last_k)
        per_sample_k = get_per_sample_distillation_k(response_mask, last_k)

    if k_batch <= 0:
        hidden_dim = repr.size(-1)
        empty = repr.new_zeros(batch_size, 0, hidden_dim)
        empty_mask = position_mask.new_zeros(batch_size, 0)
        return empty, empty_mask

    chunks_repr = []
    chunks_mask = []
    for batch_idx in range(batch_size):
        token_len = repr.size(1)
        eff_k = min(int(per_sample_k[batch_idx].item()), k_batch)
        if positions == "first_k":
            take_k = min(eff_k, token_len)
            chunk = repr[batch_idx, :take_k, :]
            chunk_mask = position_mask[batch_idx, :take_k]
        else:
            last_v = min(int(last_valid[batch_idx].item()), token_len)
            take_k = min(eff_k, last_v)
            start = last_v - take_k
            chunk = repr[batch_idx, start:last_v, :]
            chunk_mask = position_mask[batch_idx, start:last_v]
        if chunk.size(0) > k_batch:
            if positions == "first_k":
                chunk = chunk[:k_batch]
                chunk_mask = chunk_mask[:k_batch]
            else:
                chunk = chunk[-k_batch:]
                chunk_mask = chunk_mask[-k_batch:]
        pad_len = k_batch - chunk.size(0)
        if pad_len > 0:
            chunk = F.pad(chunk, (0, 0, 0, pad_len))
            chunk_mask = F.pad(chunk_mask, (0, pad_len))
        chunks_repr.append(chunk.unsqueeze(0))
        chunks_mask.append(chunk_mask.unsqueeze(0))
    return torch.cat(chunks_repr, dim=0), torch.cat(chunks_mask, dim=0)


def _compact_single_layer_position_mask(
    response_mask: torch.Tensor,
    position_mask: torch.Tensor,
    positions: str,
    last_k: int,
    first_k: int,
) -> torch.Tensor:
    """Compact ``(B, T)`` position mask to ``(B, k_batch)`` for ``first_k`` / ``last_k``."""
    _, compact_mask = _compact_single_layer_response_repr(
        position_mask.unsqueeze(-1),
        response_mask,
        position_mask,
        positions,
        last_k,
        first_k,
    )
    return compact_mask


def compact_response_repr_by_positions(
    repr: torch.Tensor,
    response_mask: torch.Tensor,
    positions: str,
    last_k: int = 32,
    first_k: int = 50,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Shrink per-token repr to the selected response positions for storage / loss.

    ``first_k`` -> ``(B, k, D)`` or ``(B, L, k, D)``; ``last_k`` -> same with width ``k``.
    ``all`` / ``last`` are returned unchanged (mask may be ``None`` for ``last``).
    """
    validate_rep_distillation_positions(positions)
    if positions in ("all", "last"):
        if positions == "all":
            return repr, response_mask
        return repr, None

    position_mask = build_rep_distillation_position_mask(
        response_mask, positions, last_k=last_k, first_k=first_k
    )
    k_target = get_compact_distillation_width(first_k if positions == "first_k" else last_k)
    token_dim = repr.size(-2) if repr.dim() == 4 else repr.size(1)
    if token_dim == k_target:
        compact_mask = build_compact_rep_distillation_position_mask(
            response_mask, positions, last_k=last_k, first_k=first_k
        )
        return repr, compact_mask

    if repr.dim() == 3:
        return _compact_single_layer_response_repr(
            repr, response_mask, position_mask, positions, last_k, first_k
        )
    if repr.dim() == 4:
        compact_layers = []
        compact_mask = None
        for layer_idx in range(repr.size(1)):
            layer_repr, layer_mask = _compact_single_layer_response_repr(
                repr[:, layer_idx],
                response_mask,
                position_mask,
                positions,
                last_k,
                first_k,
            )
            compact_layers.append(layer_repr)
            compact_mask = layer_mask
        return torch.stack(compact_layers, dim=1), compact_mask
    raise ValueError(f"Unsupported repr shape for compaction: {tuple(repr.shape)}")


def build_compact_rep_distillation_position_mask(
    response_mask: torch.Tensor,
    positions: str,
    last_k: int = 32,
    first_k: int = 50,
) -> torch.Tensor:
    """Position mask aligned with compact ``(B, k, D)`` / ``(B, L, k, D)`` repr tensors."""
    validate_rep_distillation_positions(positions)
    if positions == "first_k":
        k_batch = get_compact_distillation_width(first_k)
        if k_batch <= 0:
            return response_mask.new_zeros(response_mask.size(0), 0)
        position_mask = build_rep_distillation_position_mask(
            response_mask, positions, last_k=last_k, first_k=first_k
        )
        return _compact_single_layer_position_mask(
            response_mask, position_mask, "first_k", last_k, first_k
        )
    if positions == "last_k":
        k_batch = get_compact_distillation_width(last_k)
        if k_batch <= 0:
            return response_mask.new_zeros(response_mask.size(0), 0)
        position_mask = build_rep_distillation_position_mask(
            response_mask, positions, last_k=last_k, first_k=first_k
        )
        return _compact_single_layer_position_mask(
            response_mask, position_mask, "last_k", last_k, first_k
        )
    raise ValueError(f"build_compact_rep_distillation_position_mask does not support positions={positions!r}")


def normalized_mse_loss(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    position_mask: torch.Tensor | None = None,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """L2-normalize both vectors and compute (masked) MSE."""
    student_norm = F.normalize(student_repr, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_repr.detach(), p=2, dim=-1)

    # student_norm = student_repr
    # teacher_norm = teacher_repr.detach()

    if student_repr.dim() == 2:
        return F.mse_loss(student_norm, teacher_norm)

    per_token_mse = ((student_norm - teacher_norm) ** 2).mean(dim=-1)
    if position_mask is None:
        position_mask = torch.ones_like(per_token_mse)
    from verl.trainer.ppo.core_algos import agg_loss

    return agg_loss(per_token_mse, position_mask, loss_agg_mode)


def normalized_cosine_similarity(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean cosine similarity after L2 normalization (teacher detached)."""
    student_norm = F.normalize(student_repr, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_repr.detach(), p=2, dim=-1)

    if student_repr.dim() == 2:
        return (student_norm * teacher_norm).sum(dim=-1).mean()

    per_token_cos = (student_norm * teacher_norm).sum(dim=-1)
    if position_mask is None:
        return per_token_cos.mean()
    from verl.utils import torch_functional as verl_F

    return verl_F.masked_mean(per_token_cos, position_mask)


def _extract_single_layer_response_repr(
    hidden_states: torch.Tensor,
    response_mask: torch.Tensor,
    positions: str,
) -> torch.Tensor:
    """Per-layer repr: (B, D) for ``last``, else (B, T, D)."""
    validate_rep_distillation_positions(positions)
    if positions == "last":
        return extract_response_last_valid_token_hidden_states(hidden_states, response_mask)
    return extract_response_hidden_states(hidden_states, response_mask)


def stack_layer_response_reprs(layer_reprs: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-layer reprs into (B, L, D) or (B, L, T, D)."""
    if not layer_reprs:
        raise ValueError("layer_reprs must be non-empty")
    if len(layer_reprs) == 1:
        return layer_reprs[0]
    return torch.stack(layer_reprs, dim=1)


def get_proportional_layer_indices(
    num_target_layers: int,
    num_source_layers: int,
    *,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Map each of ``num_target_layers`` slots to an index in ``[0, num_source_layers)``.

    Used when student and teacher have different depths (e.g. 1.7B vs 4B): each student
    layer is paired with a proportionally spaced teacher layer.
    """
    if num_target_layers <= 0 or num_source_layers <= 0:
        raise ValueError("layer counts must be positive")
    if num_target_layers == num_source_layers:
        return torch.arange(num_target_layers, device=device, dtype=torch.long)
    if num_target_layers == 1:
        return torch.tensor([num_source_layers - 1], device=device, dtype=torch.long)
    if num_source_layers == 1:
        return torch.zeros(num_target_layers, device=device, dtype=torch.long)
    indices = torch.linspace(0, num_source_layers - 1, steps=num_target_layers, device=device)
    return indices.round().long()


def align_teacher_layers_to_student(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    layer_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """When layer counts differ, gather teacher layers to match student depth."""
    num_student = student.size(layer_dim)
    num_teacher = teacher.size(layer_dim)
    if num_student == num_teacher:
        return student, teacher
    indices = get_proportional_layer_indices(num_student, num_teacher, device=teacher.device)
    return student, teacher.index_select(layer_dim, indices)


def extract_teacher_response_hidden_repr(
    hidden_states: torch.Tensor | tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
    positions: str,
    layers: str = "last",
    last_k: int = 32,
    first_k: int = 50,
    compact: bool = True,
) -> torch.Tensor:
    """Extract teacher hidden repr for storage.

    Single layer (``layers='last'``):
        ``last`` -> ``(B, D)``; ``all`` -> ``(B, T, D)``;
        ``first_k``/``last_k`` -> ``(B, k, D)`` when ``compact=True``.
    Multi layer:
        ``(B, L, D)``, ``(B, L, T, D)``, or ``(B, L, k, D)`` when compact.
    """
    validate_rep_distillation_layers(layers)
    if isinstance(hidden_states, torch.Tensor):
        if layers != "last":
            raise ValueError("Multi-layer rep distillation requires the full hidden_states tuple from the model")
        repr = _extract_single_layer_response_repr(hidden_states, response_mask, positions)
    else:
        layer_indices = get_rep_distillation_hidden_state_indices(len(hidden_states), layers)
        layer_reprs = [
            _extract_single_layer_response_repr(hidden_states[idx].float(), response_mask, positions)
            for idx in layer_indices
        ]
        repr = stack_layer_response_reprs(layer_reprs)

    if compact and positions in ("first_k", "last_k"):
        repr, _ = compact_response_repr_by_positions(repr, response_mask, positions, last_k, first_k)
    return repr


def multi_layer_normalized_mse_loss(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    position_mask: torch.Tensor | None = None,
    loss_agg_mode: str = "token-mean",
    num_layers: int | None = None,
) -> torch.Tensor:
    """MSE over layers (mean). Supports (B,L,D), (B,L,T,D), and legacy 2D/3D single-layer shapes."""
    if num_layers is not None and num_layers > 1 and student_repr.dim() >= 3 and teacher_repr.dim() >= 3:
        student_repr, teacher_repr = align_teacher_layers_to_student(student_repr, teacher_repr)
        num_layers = student_repr.size(1)
    if num_layers is not None and num_layers > 1:
        if student_repr.dim() == 4:
            layer_losses = [
                normalized_mse_loss(
                    student_repr[:, layer_idx],
                    teacher_repr[:, layer_idx],
                    position_mask=position_mask,
                    loss_agg_mode=loss_agg_mode,
                )
                for layer_idx in range(num_layers)
            ]
            return torch.stack(layer_losses).mean()

        if student_repr.dim() == 3:
            layer_losses = [
                normalized_mse_loss(student_repr[:, layer_idx], teacher_repr[:, layer_idx])
                for layer_idx in range(num_layers)
            ]
            return torch.stack(layer_losses).mean()

    return normalized_mse_loss(
        student_repr,
        teacher_repr,
        position_mask=position_mask,
        loss_agg_mode=loss_agg_mode,
    )


def multi_layer_normalized_cosine_similarity(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    position_mask: torch.Tensor | None = None,
    num_layers: int | None = None,
) -> torch.Tensor:
    """Mean cosine similarity, averaged over layers when ``num_layers > 1``."""
    if num_layers is not None and num_layers > 1 and student_repr.dim() >= 3 and teacher_repr.dim() >= 3:
        student_repr, teacher_repr = align_teacher_layers_to_student(student_repr, teacher_repr)
        num_layers = student_repr.size(1)
    if num_layers is not None and num_layers > 1:
        if student_repr.dim() == 4:
            layer_sims = [
                normalized_cosine_similarity(
                    student_repr[:, layer_idx],
                    teacher_repr[:, layer_idx],
                    position_mask=position_mask,
                )
                for layer_idx in range(num_layers)
            ]
            return torch.stack(layer_sims).mean()

        if student_repr.dim() == 3:
            layer_sims = [
                normalized_cosine_similarity(student_repr[:, layer_idx], teacher_repr[:, layer_idx])
                for layer_idx in range(num_layers)
            ]
            return torch.stack(layer_sims).mean()

    return normalized_cosine_similarity(student_repr, teacher_repr, position_mask=position_mask)


def _masked_tensor_mean(tensor: torch.Tensor, position_mask: torch.Tensor | None) -> torch.Tensor:
    if position_mask is None:
        return tensor.mean()
    from verl.utils import torch_functional as verl_F

    return verl_F.masked_mean(tensor, position_mask)


def compute_rep_alignment_metrics(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    *,
    position_mask: torch.Tensor | None = None,
    num_layers: int | None = None,
) -> dict[str, float]:
    """Numeric diagnostics for rep distillation after any projector (full-rank or low-rank)."""
    aligned_student = student_repr
    aligned_teacher = teacher_repr
    aligned_layers = 1
    if num_layers is not None and num_layers > 1 and student_repr.dim() >= 3 and teacher_repr.dim() >= 3:
        aligned_student, aligned_teacher = align_teacher_layers_to_student(student_repr, teacher_repr)
        aligned_layers = int(aligned_student.size(1))

    subspace_cosine = multi_layer_normalized_cosine_similarity(
        aligned_student,
        aligned_teacher,
        position_mask=position_mask,
        num_layers=aligned_layers if aligned_layers > 1 else None,
    )
    subspace_mse = multi_layer_normalized_mse_loss(
        aligned_student,
        aligned_teacher,
        position_mask=position_mask,
        num_layers=aligned_layers if aligned_layers > 1 else None,
    )

    student_norm = aligned_student.norm(dim=-1)
    teacher_norm = aligned_teacher.norm(dim=-1)
    if aligned_student.dim() == 4:
        student_norm = student_norm.mean(dim=1)
        teacher_norm = teacher_norm.mean(dim=1)

    return {
        "rep/subspace_dim": float(aligned_student.size(-1)),
        "rep/num_aligned_layers": float(aligned_layers),
        "rep/subspace_cosine": float(subspace_cosine.detach().item()),
        "rep/subspace_mse": float(subspace_mse.detach().item()),
        "rep/subspace_student_norm_mean": float(_masked_tensor_mean(student_norm, position_mask).detach().item()),
        "rep/subspace_teacher_norm_mean": float(_masked_tensor_mean(teacher_norm, position_mask).detach().item()),
        "rep/subspace_student_norm_max": float(student_norm.max().detach().item()),
        "rep/subspace_teacher_norm_max": float(teacher_norm.max().detach().item()),
    }


def hidden_states_tuple_to_response_repr(
    hidden_states: tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
    positions: str,
    layers: str,
    *,
    indices: torch.Tensor | None = None,
    batch_size: int | None = None,
    seqlen: int | None = None,
    use_ulysses_sp: bool = False,
    pad_size: int = 0,
    ulysses_group=None,
    use_remove_padding: bool = False,
    last_k: int = 32,
    first_k: int = 50,
    compact: bool = True,
) -> torch.Tensor:
    """Build stacked response hidden repr from an HF ``hidden_states`` tuple."""
    layer_indices = get_rep_distillation_hidden_state_indices(len(hidden_states), layers)
    layer_reprs = []
    for idx in layer_indices:
        layer_hidden = hidden_states[idx]
        if use_remove_padding:
            if indices is None or batch_size is None or seqlen is None:
                raise ValueError("rmpad metadata is required when use_remove_padding=True")
            hidden_rmpad = _squeeze_hidden_rmpad(layer_hidden)
            if use_ulysses_sp:
                hidden_rmpad = gather_outputs_and_unpad(
                    hidden_rmpad,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                    group=ulysses_group,
                )
            layer_hidden = pad_input(
                hidden_states=hidden_rmpad,
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
        layer_reprs.append(
            _extract_single_layer_response_repr(layer_hidden.float(), response_mask, positions)
        )
    repr = stack_layer_response_reprs(layer_reprs)
    if compact and positions in ("first_k", "last_k"):
        repr, _ = compact_response_repr_by_positions(repr, response_mask, positions, last_k, first_k)
    return repr


def _squeeze_hidden_rmpad(last_hidden: torch.Tensor) -> torch.Tensor:
    if last_hidden.dim() == 3:
        return last_hidden.squeeze(0)
    return last_hidden


def _forward_full_last_layer_hidden(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    use_remove_padding: bool,
    use_ulysses_sp: bool = False,
    ulysses_sequence_parallel_size: int = 1,
    multi_modal_inputs: dict | None = None,
    layers: str = "last",
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run the model and return float32 hidden states (single layer or full tuple)."""
    multi_modal_inputs = multi_modal_inputs or {}
    batch_size, seqlen = input_ids.shape

    if use_remove_padding:
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        if position_ids.dim() == 3:
            position_ids_rmpad = (
                index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                .transpose(0, 1)
                .unsqueeze(1)
            )
        else:
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)

        pad_size = 0
        if use_ulysses_sp:
            is_vlm_model = hasattr(getattr(model, "module", model).config, "vision_config")
            if is_vlm_model:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                    input_ids_rmpad,
                    position_ids_rmpad=position_ids_rmpad,
                    sp_size=ulysses_sequence_parallel_size,
                )
            else:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad,
                    position_ids_rmpad=position_ids_rmpad,
                    sp_size=ulysses_sequence_parallel_size,
                )

        output = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids_rmpad,
            use_cache=False,
            output_hidden_states=True,
            **multi_modal_inputs,
        )
        if layers != "last":
            return output.hidden_states

        last_hidden_rmpad = _squeeze_hidden_rmpad(output.hidden_states[-1])

        if use_ulysses_sp:
            last_hidden_rmpad = gather_outputs_and_unpad(
                last_hidden_rmpad,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size,
            )

        return pad_input(
            hidden_states=last_hidden_rmpad,
            indices=indices,
            batch=batch_size,
            seqlen=seqlen,
        ).float()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        **multi_modal_inputs,
    )
    if layers != "last":
        return output.hidden_states
    return output.hidden_states[-1].float()


def forward_response_hidden_repr(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    use_remove_padding: bool,
    use_ulysses_sp: bool = False,
    ulysses_sequence_parallel_size: int = 1,
    multi_modal_inputs: dict | None = None,
    positions: str = "all",
    layers: str = "last",
    ulysses_group=None,
    last_k: int = 32,
    first_k: int = 50,
    compact: bool = True,
) -> torch.Tensor:
    """Forward and return response hidden repr (single- or multi-layer)."""
    validate_rep_distillation_positions(positions)
    validate_rep_distillation_layers(layers)
    with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
        hidden_output = _forward_full_last_layer_hidden(
            model,
            input_ids,
            attention_mask,
            position_ids,
            use_remove_padding=use_remove_padding,
            use_ulysses_sp=use_ulysses_sp,
            ulysses_sequence_parallel_size=ulysses_sequence_parallel_size,
            multi_modal_inputs=multi_modal_inputs,
            layers=layers,
        )

    if isinstance(hidden_output, tuple):
        batch_size, seqlen = input_ids.shape
        if use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            pad_size = 0
            if use_ulysses_sp:
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                _, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad,
                    position_ids_rmpad=index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1),
                    sp_size=ulysses_sequence_parallel_size,
                )
            else:
                indices = indices
            return hidden_states_tuple_to_response_repr(
                hidden_output,
                response_mask,
                positions,
                layers,
                indices=indices,
                batch_size=batch_size,
                seqlen=seqlen,
                use_ulysses_sp=use_ulysses_sp,
                pad_size=pad_size,
                ulysses_group=ulysses_group,
                use_remove_padding=True,
                last_k=last_k,
                first_k=first_k,
                compact=compact,
            )
        return hidden_states_tuple_to_response_repr(
            hidden_output,
            response_mask,
            positions,
            layers,
            use_remove_padding=False,
            last_k=last_k,
            first_k=first_k,
            compact=compact,
        )

    return extract_teacher_response_hidden_repr(
        hidden_output,
        response_mask,
        positions,
        layers="last",
        last_k=last_k,
        first_k=first_k,
        compact=compact,
    )


def validate_rep_projector_mode(mode: str) -> str:
    if mode not in VALID_REP_PROJECTOR_MODES:
        raise ValueError(f"rep_projector_mode must be one of {VALID_REP_PROJECTOR_MODES}, got {mode!r}")
    return mode


VALID_PS_PROJECTOR_TYPES = ("linear", "mlp", "auto")


def validate_ps_projector_type(projector_type: str) -> str:
    if projector_type not in VALID_PS_PROJECTOR_TYPES:
        raise ValueError(
            f"rep_ps_projector must be one of {VALID_PS_PROJECTOR_TYPES}, got {projector_type!r}"
        )
    return projector_type


def load_preexp_checkpoint_dict(checkpoint_path: str | Path) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def infer_subspace_mode_from_checkpoint(checkpoint: dict) -> str:
    """Return preexp2 subspace mode: ``full`` (PCA P_T), ``direct`` (trainable P_T linear), or ``residual``."""
    return str(checkpoint.get("subspace_mode", "full"))


def infer_ps_projector_type_from_checkpoint(checkpoint: dict) -> tuple[str, int | None, int | None]:
    """Return (projector_type, mlp_hidden_mult, mlp_hidden_dim)."""
    projector_type = checkpoint.get("projector_type", "linear")
    mlp_hidden_mult = checkpoint.get("mlp_hidden_mult")
    mlp_hidden_dim: int | None = None
    state_dict = checkpoint.get("state_dict", {})
    for key, weight in state_dict.items():
        if key.startswith("projectors.") and key.endswith(".0.weight"):
            projector_type = "mlp"
            mlp_hidden_dim = int(weight.shape[0])
            break
    if projector_type == "mlp" and mlp_hidden_mult is None and mlp_hidden_dim is not None:
        rank = int(checkpoint.get("rank", checkpoint.get("tail_rank", 0)) or 0)
        if rank > 0 and mlp_hidden_dim >= rank:
            mlp_hidden_mult = max(1, mlp_hidden_dim // rank)
    return projector_type, mlp_hidden_mult, mlp_hidden_dim


def resolve_student_projector_settings(
    *,
    ps_projector: str,
    mlp_hidden_mult: int,
    checkpoint_path: str | Path | None,
) -> tuple[str, int, int | None]:
    """Resolve student P_S architecture from config and optional preexp2 checkpoint."""
    ps_projector = validate_ps_projector_type(ps_projector)
    resolved_type = ps_projector
    resolved_mult = mlp_hidden_mult
    resolved_hidden_dim: int | None = None
    if checkpoint_path:
        checkpoint = load_preexp_checkpoint_dict(checkpoint_path)
        ckpt_type, ckpt_mult, ckpt_hidden = infer_ps_projector_type_from_checkpoint(checkpoint)
        if ps_projector == "auto":
            resolved_type = ckpt_type
        elif ps_projector != ckpt_type:
            raise ValueError(
                f"rep_ps_projector={ps_projector!r} but checkpoint {checkpoint_path} "
                f"has projector_type={ckpt_type!r}"
            )
        if ckpt_type == "mlp":
            if ckpt_mult is not None:
                resolved_mult = int(ckpt_mult)
            resolved_hidden_dim = ckpt_hidden
    if resolved_type == "auto":
        resolved_type = "linear"
    return resolved_type, resolved_mult, resolved_hidden_dim


def build_student_subspace_projector(
    student_dim: int,
    rank: int,
    *,
    projector_type: str,
    mlp_hidden_mult: int = 4,
    mlp_hidden_dim: int | None = None,
) -> nn.Module:
    if projector_type == "linear":
        return nn.Linear(student_dim, rank, bias=False)
    if projector_type == "mlp":
        hidden_dim = mlp_hidden_dim or max(rank, mlp_hidden_mult * rank)
        return nn.Sequential(
            nn.Linear(student_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, rank, bias=False),
        )
    raise ValueError(f"Unsupported student projector_type={projector_type!r}")


def student_projector_weight_norm(projector: nn.Module) -> float:
    if isinstance(projector, nn.Linear):
        return float(projector.weight.detach().float().norm().item())
    if isinstance(projector, nn.Sequential):
        return float(
            sum(param.detach().float().norm().item() for param in projector.parameters())
        )
    raise TypeError(f"Unsupported student projector module: {type(projector)!r}")


def load_student_projector_from_state_dict(
    projector: nn.Module,
    state_dict: dict[str, Tensor],
    layer_key: str,
    *,
    rank: int,
    projector_type: str,
) -> bool:
    if projector_type == "linear":
        student_key = f"projectors.{layer_key}.weight"
        if student_key not in state_dict:
            return False
        weight = state_dict[student_key]
        out_rank = min(weight.shape[0], rank)
        projector.weight.data[:out_rank].copy_(weight[:out_rank])
        return True

    first_key = f"projectors.{layer_key}.0.weight"
    last_key = f"projectors.{layer_key}.2.weight"
    if first_key not in state_dict or last_key not in state_dict:
        return False
    if not isinstance(projector, nn.Sequential):
        raise TypeError("Expected nn.Sequential student projector for MLP checkpoint")
    projector[0].weight.data.copy_(state_dict[first_key])
    weight = state_dict[last_key]
    out_rank = min(weight.shape[0], rank)
    projector[2].weight.data[:out_rank].copy_(weight[:out_rank])
    return True


@torch.no_grad()
def fit_teacher_pca_from_rows(h_teacher: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Return (P_T, mean) with P_T shape (rank, dim), equivalent to PCA/SVD on centered rows."""
    if h_teacher.shape[0] < 2:
        raise ValueError("Need at least 2 teacher rows to fit PCA")
    mean = h_teacher.mean(dim=0)
    centered = h_teacher - mean
    n = centered.shape[0]
    cov = (centered.T @ centered) / max(n - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    components = eigvecs[:, order].T[:rank]
    return components.float(), mean.float()


class LowRankCrossArchProjector(nn.Module):
    """Frozen teacher PCA projectors P_T and trainable student projectors P_S.

    Distills in a shared r-dimensional subspace:
        z_S = P_S h_S,  z_T = P_T (h_T - mu_T),  L = ||z_S - sg(z_T)||^2
    """

    def __init__(
        self,
        *,
        num_layers: int,
        student_dim: int,
        teacher_dim: int,
        rank: int,
        num_teacher_layers: int,
        ps_projector_type: str = "linear",
        mlp_hidden_mult: int = 4,
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.rank = rank
        self.num_teacher_layers = num_teacher_layers
        self.ps_projector_type = validate_ps_projector_type(ps_projector_type)
        if self.ps_projector_type == "auto":
            self.ps_projector_type = "linear"
        self.mlp_hidden_mult = mlp_hidden_mult
        self.teacher_layer_indices = get_proportional_layer_indices(
            num_layers, num_teacher_layers
        ).tolist()

        self.student_projectors = nn.ModuleList(
            [
                build_student_subspace_projector(
                    student_dim,
                    rank,
                    projector_type=self.ps_projector_type,
                    mlp_hidden_mult=mlp_hidden_mult,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer(
            "teacher_weights",
            torch.zeros(num_layers, rank, teacher_dim),
            persistent=True,
        )
        self.register_buffer(
            "teacher_means",
            torch.zeros(num_layers, teacher_dim),
            persistent=True,
        )
        self._teacher_pt_initialized = False
        self._loaded_from_checkpoint = False
        self._loaded_ps_layers = 0
        self._loaded_pt_layers = 0
        self._ps_frozen = False

    def freeze_student_projectors(self) -> None:
        """Freeze P_S; gradients still flow through to the student backbone."""
        for param in self.student_projectors.parameters():
            param.requires_grad_(False)
        self._ps_frozen = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self._ps_frozen:
            return []
        return list(self.student_projectors.parameters())

    def project_student(self, hidden: Tensor, layer_idx: int) -> Tensor:
        return self.student_projectors[layer_idx](hidden)

    def project_teacher(self, hidden: Tensor, layer_idx: int) -> Tensor:
        centered = hidden - self.teacher_means[layer_idx].to(hidden.dtype)
        weight = self.teacher_weights[layer_idx].to(hidden.dtype)
        return centered @ weight.T

    @torch.no_grad()
    def projector_param_metrics(self) -> dict[str, float]:
        """Direct diagnostics for trainable P_S and frozen P_T parameters."""
        ps_norms = torch.tensor(
            [
                student_projector_weight_norm(proj)
                for proj in self.student_projectors
            ],
            dtype=torch.float32,
        )
        pt_weight_norms = self.teacher_weights.detach().float().norm(dim=(1, 2))
        pt_mean_norms = self.teacher_means.detach().float().norm(dim=1)
        return {
            "rep/ps_weight_norm_mean": float(ps_norms.mean().item()),
            "rep/ps_weight_norm_max": float(ps_norms.max().item()),
            "rep/pt_weight_norm_mean": float(pt_weight_norms.mean().item()),
            "rep/pt_weight_norm_max": float(pt_weight_norms.max().item()),
            "rep/pt_mean_norm_mean": float(pt_mean_norms.mean().item()),
            "rep/pt_mean_norm_max": float(pt_mean_norms.max().item()),
            "rep/ps_projector_type_mlp": float(self.ps_projector_type == "mlp"),
            "rep/mlp_hidden_mult": float(self.mlp_hidden_mult),
            "rep/teacher_pt_initialized": float(self._teacher_pt_initialized),
            "rep/projector_loaded_from_checkpoint": float(self._loaded_from_checkpoint),
            "rep/ps_layers_loaded": float(self._loaded_ps_layers),
            "rep/pt_layers_loaded": float(self._loaded_pt_layers),
            "rep/ps_frozen": float(self._ps_frozen),
            "rep/subspace_mode_direct": 0.0,
        }

    @torch.no_grad()
    def maybe_init_teacher_pca_from_batch(self, teacher_repr: Tensor, *, num_layers: int) -> None:
        if self._teacher_pt_initialized:
            return
        if num_layers > 1 and teacher_repr.dim() >= 3 and teacher_repr.size(1) != num_layers:
            indices = get_proportional_layer_indices(
                num_layers, teacher_repr.size(1), device=teacher_repr.device
            )
            teacher_repr = teacher_repr.index_select(1, indices)
        for layer_idx in range(num_layers):
            rows = self._flatten_hidden_rows(teacher_repr, layer_idx, num_layers)
            rank = min(self.rank, rows.shape[0] - 1, rows.shape[1])
            if rank < 1:
                raise ValueError(f"Cannot fit teacher PCA for layer {layer_idx}: not enough rows")
            weight, mean = fit_teacher_pca_from_rows(rows, rank)
            self.teacher_weights[layer_idx, :rank].copy_(weight)
            self.teacher_means[layer_idx].copy_(mean)
        self._teacher_pt_initialized = True

    def project_pair(
        self,
        student_repr: Tensor,
        teacher_repr: Tensor,
        *,
        num_layers: int | None,
    ) -> tuple[Tensor, Tensor]:
        if num_layers is not None and num_layers > 1:
            student_repr, teacher_repr = align_teacher_layers_to_student(student_repr, teacher_repr)
            student_layers = [
                self.project_student(student_repr.select(1, layer_idx), layer_idx)
                for layer_idx in range(num_layers)
            ]
            teacher_layers = [
                self.project_teacher(teacher_repr.select(1, layer_idx), layer_idx)
                for layer_idx in range(num_layers)
            ]
            return torch.stack(student_layers, dim=1), torch.stack(teacher_layers, dim=1)
        return self.project_student(student_repr, 0), self.project_teacher(teacher_repr, 0)

    @staticmethod
    def _flatten_hidden_rows(repr_tensor: Tensor, layer_idx: int, num_layers: int) -> Tensor:
        layer_hidden = repr_tensor.select(1, layer_idx) if num_layers > 1 else repr_tensor
        if layer_hidden.dim() == 2:
            return layer_hidden.reshape(-1, layer_hidden.shape[-1]).float()
        if layer_hidden.dim() == 3:
            return layer_hidden.reshape(-1, layer_hidden.shape[-1]).float()
        raise ValueError(f"Unsupported teacher repr shape for PCA init: {tuple(repr_tensor.shape)}")

    @staticmethod
    def _checkpoint_layer_keys(student_layer: int, teacher_layer: int) -> list[str]:
        return [
            f"s{student_layer}_t{teacher_layer}",
            f"{student_layer}_{teacher_layer}",
        ]

    def load_from_preexp_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_preexp_checkpoint_dict(checkpoint_path)
        ckpt_type, ckpt_mult, ckpt_hidden = infer_ps_projector_type_from_checkpoint(checkpoint)
        if self.ps_projector_type != ckpt_type:
            raise ValueError(
                f"LowRankCrossArchProjector expects ps_projector_type={self.ps_projector_type!r}, "
                f"but checkpoint {checkpoint_path} has projector_type={ckpt_type!r}"
            )
        if ckpt_type == "mlp" and ckpt_hidden is not None:
            expected_hidden = ckpt_hidden
            actual_hidden = self.student_projectors[0][0].weight.shape[0]
            if actual_hidden != expected_hidden:
                raise ValueError(
                    f"MLP hidden dim mismatch: projector has {actual_hidden}, "
                    f"checkpoint {checkpoint_path} has {expected_hidden}"
                )
        state_dict = checkpoint.get("state_dict", {})
        frozen_weights = checkpoint.get("frozen_pt_weights", {})
        frozen_means = checkpoint.get("frozen_pt_means", {})

        loaded_ps_layers = 0
        loaded_pt_layers = 0
        for layer_idx in range(self.num_layers):
            teacher_layer = self.teacher_layer_indices[layer_idx]
            for key in self._checkpoint_layer_keys(layer_idx, teacher_layer):
                if load_student_projector_from_state_dict(
                    self.student_projectors[layer_idx],
                    state_dict,
                    key,
                    rank=self.rank,
                    projector_type=self.ps_projector_type,
                ):
                    loaded_ps_layers += 1
                    break
            for key in self._checkpoint_layer_keys(layer_idx, teacher_layer):
                if key in frozen_weights:
                    weight = frozen_weights[key]
                    rank = min(weight.shape[0], self.rank)
                    self.teacher_weights[layer_idx, :rank].copy_(weight[:rank])
                    loaded_pt_layers += 1
                if key in frozen_means:
                    self.teacher_means[layer_idx].copy_(frozen_means[key])
                if key in frozen_weights:
                    break

        self._loaded_ps_layers = loaded_ps_layers
        self._loaded_pt_layers = loaded_pt_layers
        self._loaded_from_checkpoint = loaded_ps_layers > 0 or loaded_pt_layers > 0
        self._teacher_pt_initialized = loaded_pt_layers == self.num_layers and (
            float(self.teacher_weights.abs().sum().item()) > 0.0
        )
        if loaded_ps_layers > 0 and loaded_pt_layers == 0 and frozen_weights:
            raise ValueError(
                f"P_S loaded from {checkpoint_path} but P_T keys did not match. "
                f"frozen_pt_weights keys sample: {list(frozen_weights.keys())[:5]}. "
                "Re-save ps_bank.pt with preexp2 or delete REP_LOW_RANK_INIT_CHECKPOINT to use PCA init."
            )


class DirectLowRankCrossArchProjector(nn.Module):
    """Frozen linear P_T and trainable/frozen linear P_S from direct preexp2 (no teacher PCA).

    Offline direct preexp2 learns:
        L = ||P_S h_S - P_T h_T||^2
    Online OPD keeps both projectors fixed by default (P_T always frozen; P_S optional via rep_freeze_ps).
    """

    def __init__(
        self,
        *,
        num_layers: int,
        student_dim: int,
        teacher_dim: int,
        rank: int,
        num_teacher_layers: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.rank = rank
        self.num_teacher_layers = num_teacher_layers
        self.ps_projector_type = "linear"
        self.mlp_hidden_mult = 4
        self.teacher_layer_indices = get_proportional_layer_indices(
            num_layers, num_teacher_layers
        ).tolist()

        self.student_projectors = nn.ModuleList(
            [
                build_student_subspace_projector(
                    student_dim,
                    rank,
                    projector_type="linear",
                )
                for _ in range(num_layers)
            ]
        )
        self.teacher_projectors = nn.ModuleList(
            [nn.Linear(teacher_dim, rank, bias=False) for _ in range(num_layers)]
        )
        for param in self.teacher_projectors.parameters():
            param.requires_grad_(False)

        self._teacher_pt_initialized = False
        self._loaded_from_checkpoint = False
        self._loaded_ps_layers = 0
        self._loaded_pt_layers = 0
        self._ps_frozen = False

    def freeze_student_projectors(self) -> None:
        for param in self.student_projectors.parameters():
            param.requires_grad_(False)
        self._ps_frozen = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self._ps_frozen:
            return []
        return list(self.student_projectors.parameters())

    def project_student(self, hidden: Tensor, layer_idx: int) -> Tensor:
        return self.student_projectors[layer_idx](hidden)

    def project_teacher(self, hidden: Tensor, layer_idx: int) -> Tensor:
        return self.teacher_projectors[layer_idx](hidden)

    @torch.no_grad()
    def projector_param_metrics(self) -> dict[str, float]:
        ps_norms = torch.tensor(
            [student_projector_weight_norm(proj) for proj in self.student_projectors],
            dtype=torch.float32,
        )
        pt_weight_norms = torch.tensor(
            [proj.weight.detach().float().norm().item() for proj in self.teacher_projectors],
            dtype=torch.float32,
        )
        return {
            "rep/ps_weight_norm_mean": float(ps_norms.mean().item()),
            "rep/ps_weight_norm_max": float(ps_norms.max().item()),
            "rep/pt_weight_norm_mean": float(pt_weight_norms.mean().item()),
            "rep/pt_weight_norm_max": float(pt_weight_norms.max().item()),
            "rep/pt_mean_norm_mean": 0.0,
            "rep/pt_mean_norm_max": 0.0,
            "rep/ps_projector_type_mlp": 0.0,
            "rep/mlp_hidden_mult": float(self.mlp_hidden_mult),
            "rep/teacher_pt_initialized": float(self._teacher_pt_initialized),
            "rep/projector_loaded_from_checkpoint": float(self._loaded_from_checkpoint),
            "rep/ps_layers_loaded": float(self._loaded_ps_layers),
            "rep/pt_layers_loaded": float(self._loaded_pt_layers),
            "rep/ps_frozen": float(self._ps_frozen),
            "rep/subspace_mode_direct": 1.0,
        }

    @torch.no_grad()
    def maybe_init_teacher_pca_from_batch(self, teacher_repr: Tensor, *, num_layers: int) -> None:
        if self._teacher_pt_initialized:
            return
        raise ValueError(
            "Direct low-rank projector requires P_T from a direct preexp2 checkpoint "
            "(subspace_mode=direct, direct_bank.pt). Set rep_low_rank_init_checkpoint or "
            "train with --subspace-mode direct first."
        )

    def project_pair(
        self,
        student_repr: Tensor,
        teacher_repr: Tensor,
        *,
        num_layers: int | None,
    ) -> tuple[Tensor, Tensor]:
        if num_layers is not None and num_layers > 1:
            student_repr, teacher_repr = align_teacher_layers_to_student(student_repr, teacher_repr)
            student_layers = [
                self.project_student(student_repr.select(1, layer_idx), layer_idx)
                for layer_idx in range(num_layers)
            ]
            teacher_layers = [
                self.project_teacher(teacher_repr.select(1, layer_idx), layer_idx)
                for layer_idx in range(num_layers)
            ]
            return torch.stack(student_layers, dim=1), torch.stack(teacher_layers, dim=1)
        return self.project_student(student_repr, 0), self.project_teacher(teacher_repr, 0)

    def load_from_preexp_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_preexp_checkpoint_dict(checkpoint_path)
        if infer_subspace_mode_from_checkpoint(checkpoint) != "direct":
            raise ValueError(
                f"DirectLowRankCrossArchProjector expected subspace_mode=direct in {checkpoint_path}, "
                f"got {checkpoint.get('subspace_mode')!r}"
            )
        ckpt_type, _, _ = infer_ps_projector_type_from_checkpoint(checkpoint)
        if ckpt_type != "linear":
            raise ValueError(
                f"Direct preexp2 checkpoint must use linear projectors, got projector_type={ckpt_type!r}"
            )

        state_dict = checkpoint.get("state_dict", {})
        pt_state_dict = checkpoint.get("pt_state_dict", {})
        if not pt_state_dict:
            raise ValueError(
                f"Direct checkpoint {checkpoint_path} is missing pt_state_dict. "
                "Re-run preexp2 with --subspace-mode direct."
            )

        loaded_ps_layers = 0
        loaded_pt_layers = 0
        for layer_idx in range(self.num_layers):
            teacher_layer = self.teacher_layer_indices[layer_idx]
            for key in LowRankCrossArchProjector._checkpoint_layer_keys(layer_idx, teacher_layer):
                if load_student_projector_from_state_dict(
                    self.student_projectors[layer_idx],
                    state_dict,
                    key,
                    rank=self.rank,
                    projector_type="linear",
                ):
                    loaded_ps_layers += 1
                    break
            for key in LowRankCrossArchProjector._checkpoint_layer_keys(layer_idx, teacher_layer):
                if load_student_projector_from_state_dict(
                    self.teacher_projectors[layer_idx],
                    pt_state_dict,
                    key,
                    rank=self.rank,
                    projector_type="linear",
                ):
                    loaded_pt_layers += 1
                    break

        self._loaded_ps_layers = loaded_ps_layers
        self._loaded_pt_layers = loaded_pt_layers
        self._loaded_from_checkpoint = loaded_ps_layers > 0 or loaded_pt_layers > 0
        self._teacher_pt_initialized = loaded_pt_layers == self.num_layers and (
            float(
                sum(proj.weight.detach().abs().sum().item() for proj in self.teacher_projectors)
            )
            > 0.0
        )
        if loaded_ps_layers > 0 and loaded_pt_layers == 0:
            raise ValueError(
                f"P_S loaded from {checkpoint_path} but P_T keys did not match pt_state_dict. "
                f"pt_state_dict keys sample: {list(pt_state_dict.keys())[:5]}."
            )


def subtract_rowspace_projection(
    hidden: Tensor,
    weight: Tensor,
    *,
    mean: Tensor | None = None,
) -> Tensor:
    """Remove the component in row(weight) from hidden.

    weight: (r, d), hidden: (..., d). If mean is set, center hidden first (teacher head).
    Returns residual (..., d) orthogonal to row(weight) when rows are orthonormal.
    """
    orig_shape = hidden.shape
    flat = hidden.reshape(-1, hidden.shape[-1])
    if mean is not None:
        flat = flat - mean.to(device=flat.device, dtype=flat.dtype).unsqueeze(0)
    weight = weight.to(device=flat.device, dtype=flat.dtype)
    coords = flat @ weight.T
    parallel = coords @ weight
    residual = flat - parallel
    return residual.reshape(orig_shape)


def _load_preexp_checkpoint_tensors(
    checkpoint_path: str | Path,
    *,
    num_layers: int,
    teacher_layer_indices: list[int],
    rank: int,
) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], int, int]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    frozen_weights = checkpoint.get("frozen_pt_weights", {})
    frozen_means = checkpoint.get("frozen_pt_means", {})

    ps_by_layer: dict[int, Tensor] = {}
    pt_by_layer: dict[int, Tensor] = {}
    mean_by_layer: dict[int, Tensor] = {}
    loaded_ps = 0
    loaded_pt = 0
    for layer_idx in range(num_layers):
        teacher_layer = teacher_layer_indices[layer_idx]
        keys = LowRankCrossArchProjector._checkpoint_layer_keys(layer_idx, teacher_layer)
        for key in keys:
            student_key = f"projectors.{key}.weight"
            if student_key in state_dict:
                ps_by_layer[layer_idx] = state_dict[student_key][:rank].clone()
                loaded_ps += 1
                break
        for key in keys:
            if key in frozen_weights:
                pt_by_layer[layer_idx] = frozen_weights[key][:rank].clone()
                loaded_pt += 1
            if key in frozen_means:
                mean_by_layer[layer_idx] = frozen_means[key].clone()
            if key in frozen_weights:
                break
    return ps_by_layer, pt_by_layer, mean_by_layer, loaded_ps, loaded_pt


class ResidualLowRankCrossArchProjector(nn.Module):
    """Distill in the orthogonal complement of a frozen head subspace.

    Head (frozen): top-r_h PCA bridge from pre-experiment 2.
    Tail: z = P_tail h_residual, where h_res = h - proj_head(h).
    """

    def __init__(
        self,
        *,
        num_layers: int,
        student_dim: int,
        teacher_dim: int,
        head_rank: int,
        tail_rank: int,
        num_teacher_layers: int,
        tail_ps_projector_type: str = "linear",
        tail_mlp_hidden_mult: int = 4,
        tail_mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if head_rank <= 0 or tail_rank <= 0:
            raise ValueError("head_rank and tail_rank must be positive")
        self.num_layers = num_layers
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.head_rank = head_rank
        self.tail_rank = tail_rank
        self.rank = tail_rank
        self.tail_ps_projector_type = validate_ps_projector_type(tail_ps_projector_type)
        if self.tail_ps_projector_type == "auto":
            self.tail_ps_projector_type = "linear"
        self.tail_mlp_hidden_mult = tail_mlp_hidden_mult
        self.num_teacher_layers = num_teacher_layers
        self.teacher_layer_indices = get_proportional_layer_indices(
            num_layers, num_teacher_layers
        ).tolist()

        self.register_buffer(
            "head_student_weights",
            torch.zeros(num_layers, head_rank, student_dim),
            persistent=True,
        )
        self.register_buffer(
            "head_teacher_weights",
            torch.zeros(num_layers, head_rank, teacher_dim),
            persistent=True,
        )
        self.register_buffer(
            "head_teacher_means",
            torch.zeros(num_layers, teacher_dim),
            persistent=True,
        )
        self.tail_student_projectors = nn.ModuleList(
            [
                build_student_subspace_projector(
                    student_dim,
                    tail_rank,
                    projector_type=self.tail_ps_projector_type,
                    mlp_hidden_mult=tail_mlp_hidden_mult,
                    mlp_hidden_dim=tail_mlp_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer(
            "tail_teacher_weights",
            torch.zeros(num_layers, tail_rank, teacher_dim),
            persistent=True,
        )
        self.register_buffer(
            "tail_teacher_means",
            torch.zeros(num_layers, teacher_dim),
            persistent=True,
        )
        self._head_loaded = False
        self._tail_teacher_initialized = False
        self._loaded_tail_ps_layers = 0
        self._loaded_tail_pt_layers = 0
        self._loaded_from_checkpoint = False
        self._ps_frozen = False

    def freeze_student_projectors(self) -> None:
        for param in self.tail_student_projectors.parameters():
            param.requires_grad_(False)
        self._ps_frozen = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self._ps_frozen:
            return []
        return list(self.tail_student_projectors.parameters())

    def load_head_from_preexp_checkpoint(self, checkpoint_path: str | Path) -> None:
        ps_by_layer, pt_by_layer, mean_by_layer, loaded_ps, loaded_pt = _load_preexp_checkpoint_tensors(
            checkpoint_path,
            num_layers=self.num_layers,
            teacher_layer_indices=self.teacher_layer_indices,
            rank=self.head_rank,
        )
        if loaded_pt != self.num_layers:
            raise ValueError(
                f"Head P_T: expected {self.num_layers} layers from {checkpoint_path}, loaded {loaded_pt}"
            )
        for layer_idx in range(self.num_layers):
            if layer_idx in pt_by_layer:
                self.head_teacher_weights[layer_idx].copy_(pt_by_layer[layer_idx])
            if layer_idx in mean_by_layer:
                self.head_teacher_means[layer_idx].copy_(mean_by_layer[layer_idx])
            if layer_idx in ps_by_layer:
                self.head_student_weights[layer_idx].copy_(ps_by_layer[layer_idx])
        self._head_loaded = loaded_pt == self.num_layers

    def load_tail_from_preexp_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_preexp_checkpoint_dict(checkpoint_path)
        ckpt_type, ckpt_mult, ckpt_hidden = infer_ps_projector_type_from_checkpoint(checkpoint)
        if self.tail_ps_projector_type != ckpt_type:
            raise ValueError(
                f"ResidualLowRankCrossArchProjector expects tail_ps_projector_type="
                f"{self.tail_ps_projector_type!r}, but checkpoint {checkpoint_path} "
                f"has projector_type={ckpt_type!r}"
            )
        if ckpt_type == "mlp" and ckpt_hidden is not None:
            actual_hidden = self.tail_student_projectors[0][0].weight.shape[0]
            if actual_hidden != ckpt_hidden:
                raise ValueError(
                    f"Tail MLP hidden dim mismatch: projector has {actual_hidden}, "
                    f"checkpoint {checkpoint_path} has {ckpt_hidden}"
                )
        state_dict = checkpoint.get("state_dict", checkpoint.get("tail_state_dict", {}))
        frozen_weights = checkpoint.get(
            "frozen_tail_pt_weights", checkpoint.get("frozen_pt_weights", {})
        )
        frozen_means = checkpoint.get("frozen_tail_pt_means", checkpoint.get("frozen_pt_means", {}))

        loaded_ps = 0
        loaded_pt = 0
        for layer_idx in range(self.num_layers):
            teacher_layer = self.teacher_layer_indices[layer_idx]
            for key in LowRankCrossArchProjector._checkpoint_layer_keys(layer_idx, teacher_layer):
                if load_student_projector_from_state_dict(
                    self.tail_student_projectors[layer_idx],
                    state_dict,
                    key,
                    rank=self.tail_rank,
                    projector_type=self.tail_ps_projector_type,
                ):
                    loaded_ps += 1
                    break
            for key in LowRankCrossArchProjector._checkpoint_layer_keys(layer_idx, teacher_layer):
                if key in frozen_weights:
                    weight = frozen_weights[key]
                    rank = min(weight.shape[0], self.tail_rank)
                    self.tail_teacher_weights[layer_idx, :rank].copy_(weight[:rank])
                    loaded_pt += 1
                if key in frozen_means:
                    self.tail_teacher_means[layer_idx].copy_(frozen_means[key])
                if key in frozen_weights:
                    break

        self._loaded_tail_ps_layers = loaded_ps
        self._loaded_tail_pt_layers = loaded_pt
        self._loaded_from_checkpoint = loaded_ps > 0 or loaded_pt > 0
        self._tail_teacher_initialized = loaded_pt == self.num_layers and (
            float(self.tail_teacher_weights.abs().sum().item()) > 0.0
        )

    def _student_head_residual(self, hidden: Tensor, layer_idx: int) -> Tensor:
        weight = self.head_student_weights[layer_idx]
        return subtract_rowspace_projection(hidden, weight)

    def _teacher_head_residual(self, hidden: Tensor, layer_idx: int) -> Tensor:
        weight = self.head_teacher_weights[layer_idx]
        mean = self.head_teacher_means[layer_idx]
        return subtract_rowspace_projection(hidden, weight, mean=mean)

    def project_tail_student(self, hidden: Tensor, layer_idx: int) -> Tensor:
        return self.tail_student_projectors[layer_idx](hidden)

    def project_tail_teacher(self, hidden: Tensor, layer_idx: int) -> Tensor:
        centered = hidden - self.tail_teacher_means[layer_idx].to(hidden.dtype)
        weight = self.tail_teacher_weights[layer_idx].to(hidden.dtype)
        return centered @ weight.T

    @torch.no_grad()
    def head_alignment_metrics(
        self,
        student_repr: Tensor,
        teacher_repr: Tensor,
        *,
        num_layers: int | None,
    ) -> dict[str, float]:
        if num_layers is None or num_layers <= 1:
            z_s = student_repr @ self.head_student_weights[0].T
            h_t = teacher_repr - self.head_teacher_means[0].to(teacher_repr.dtype)
            z_t = h_t @ self.head_teacher_weights[0].T
            cos = multi_layer_normalized_cosine_similarity(z_s.unsqueeze(1), z_t.unsqueeze(1), num_layers=1)
            return {"rep/head_subspace_cosine": float(cos.detach().item())}

        student_repr, teacher_repr = align_teacher_layers_to_student(student_repr, teacher_repr)
        head_cosines = []
        for layer_idx in range(num_layers):
            h_s = student_repr.select(1, layer_idx)
            h_t = teacher_repr.select(1, layer_idx)
            z_s = h_s @ self.head_student_weights[layer_idx].T
            h_t_c = h_t - self.head_teacher_means[layer_idx].to(h_t.dtype)
            z_t = h_t_c @ self.head_teacher_weights[layer_idx].T
            head_cosines.append(
                multi_layer_normalized_cosine_similarity(
                    z_s.unsqueeze(1), z_t.unsqueeze(1), num_layers=1
                )
            )
        head_cos = torch.stack(head_cosines).mean()
        return {"rep/head_subspace_cosine": float(head_cos.detach().item())}

    @torch.no_grad()
    def projector_param_metrics(self) -> dict[str, float]:
        tail_ps_norms = torch.tensor(
            [
                student_projector_weight_norm(proj)
                for proj in self.tail_student_projectors
            ],
            dtype=torch.float32,
        )
        tail_pt_norms = self.tail_teacher_weights.detach().float().norm(dim=(1, 2))
        head_ps_norms = self.head_student_weights.detach().float().norm(dim=(1, 2))
        head_pt_norms = self.head_teacher_weights.detach().float().norm(dim=(1, 2))
        return {
            "rep/ps_weight_norm_mean": float(tail_ps_norms.mean().item()),
            "rep/ps_weight_norm_max": float(tail_ps_norms.max().item()),
            "rep/pt_weight_norm_mean": float(tail_pt_norms.mean().item()),
            "rep/pt_weight_norm_max": float(tail_pt_norms.max().item()),
            "rep/head_ps_weight_norm_mean": float(head_ps_norms.mean().item()),
            "rep/head_pt_weight_norm_mean": float(head_pt_norms.mean().item()),
            "rep/ps_projector_type_mlp": float(self.tail_ps_projector_type == "mlp"),
            "rep/mlp_hidden_mult": float(self.tail_mlp_hidden_mult),
            "rep/teacher_pt_initialized": float(self._tail_teacher_initialized),
            "rep/projector_loaded_from_checkpoint": float(self._loaded_from_checkpoint),
            "rep/ps_layers_loaded": float(self._loaded_tail_ps_layers),
            "rep/pt_layers_loaded": float(self._loaded_tail_pt_layers),
            "rep/ps_frozen": float(self._ps_frozen),
            "rep/use_residual_projector": 1.0,
            "rep/head_rank": float(self.head_rank),
            "rep/tail_rank": float(self.tail_rank),
            "rep/head_loaded": float(self._head_loaded),
        }

    @torch.no_grad()
    def maybe_init_tail_pca_from_batch(self, teacher_repr: Tensor, *, num_layers: int) -> None:
        if self._tail_teacher_initialized:
            return
        if not self._head_loaded:
            raise RuntimeError("Head projectors must be loaded before tail PCA init")
        if num_layers > 1 and teacher_repr.dim() >= 3 and teacher_repr.size(1) != num_layers:
            indices = get_proportional_layer_indices(
                num_layers, teacher_repr.size(1), device=teacher_repr.device
            )
            teacher_repr = teacher_repr.index_select(1, indices)
        for layer_idx in range(num_layers):
            rows = LowRankCrossArchProjector._flatten_hidden_rows(teacher_repr, layer_idx, num_layers)
            head_weight = self.head_teacher_weights[layer_idx]
            head_mean = self.head_teacher_means[layer_idx]
            residual_rows = subtract_rowspace_projection(rows, head_weight, mean=head_mean)
            rank = min(self.tail_rank, residual_rows.shape[0] - 1, residual_rows.shape[1])
            if rank < 1:
                raise ValueError(f"Cannot fit tail PCA for layer {layer_idx}: not enough rows")
            weight, mean = fit_teacher_pca_from_rows(residual_rows, rank)
            self.tail_teacher_weights[layer_idx, :rank].copy_(weight)
            self.tail_teacher_means[layer_idx].copy_(mean)
        self._tail_teacher_initialized = True

    def project_pair(
        self,
        student_repr: Tensor,
        teacher_repr: Tensor,
        *,
        num_layers: int | None,
    ) -> tuple[Tensor, Tensor]:
        if not self._head_loaded:
            raise RuntimeError("Residual projector requires head checkpoint (rep_head_init_checkpoint)")
        if num_layers is not None and num_layers > 1:
            student_repr, teacher_repr = align_teacher_layers_to_student(student_repr, teacher_repr)
            student_layers = []
            teacher_layers = []
            for layer_idx in range(num_layers):
                h_s_res = self._student_head_residual(student_repr.select(1, layer_idx), layer_idx)
                h_t_res = self._teacher_head_residual(teacher_repr.select(1, layer_idx), layer_idx)
                student_layers.append(self.project_tail_student(h_s_res, layer_idx))
                teacher_layers.append(self.project_tail_teacher(h_t_res, layer_idx))
            return torch.stack(student_layers, dim=1), torch.stack(teacher_layers, dim=1)

        h_s_res = self._student_head_residual(student_repr, 0)
        h_t_res = self._teacher_head_residual(teacher_repr, 0)
        return self.project_tail_student(h_s_res, 0), self.project_tail_teacher(h_t_res, 0)
