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

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_name
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs

RepDistillationPositions = Literal["last", "all", "last_k", "first_k"]
VALID_REP_DISTILLATION_POSITIONS = ("last", "all", "last_k", "first_k")

RepDistillationLayers = Literal["last", "all", "even", "odd"]
VALID_REP_DISTILLATION_LAYERS = ("last", "all", "even", "odd")


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
    # student_norm = F.normalize(student_repr, p=2, dim=-1)
    # teacher_norm = F.normalize(teacher_repr.detach(), p=2, dim=-1)

    student_norm = student_repr
    teacher_norm = teacher_repr.detach()

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
