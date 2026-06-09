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

import torch

from verl.utils.rep_distillation import (
    align_teacher_layers_to_student,
    build_compact_rep_distillation_position_mask,
    build_rep_distillation_position_mask,
    compact_response_repr_by_positions,
    get_batch_distillation_k,
    get_proportional_layer_indices,
    multi_layer_normalized_mse_loss,
)


def test_last_k_uses_all_tokens_when_response_shorter_than_k():
    response_len = 128
    last_k = 25
    valid_len = 10
    response_mask = torch.zeros(2, response_len, dtype=torch.float32)
    response_mask[:, :valid_len] = 1.0

    position_mask = build_rep_distillation_position_mask(response_mask, "last_k", last_k=last_k)
    assert int(position_mask[0].sum().item()) == valid_len

    repr_tensor = torch.randn(2, response_len, 8)
    compact_repr, compact_mask = compact_response_repr_by_positions(
        repr_tensor, response_mask, "last_k", last_k=last_k
    )
    assert compact_repr.shape == (2, last_k, 8)
    assert int(compact_mask[0].sum().item()) == valid_len
    assert torch.allclose(compact_repr[0, :valid_len], repr_tensor[0, :valid_len])


def test_first_k_uses_all_tokens_when_response_shorter_than_k():
    response_len = 128
    first_k = 25
    valid_len = 10
    response_mask = torch.zeros(1, response_len, dtype=torch.float32)
    response_mask[:, :valid_len] = 1.0

    position_mask = build_rep_distillation_position_mask(response_mask, "first_k", first_k=first_k)
    assert int(position_mask[0].sum().item()) == valid_len

    repr_tensor = torch.randn(1, response_len, 8)
    compact_repr, compact_mask = compact_response_repr_by_positions(
        repr_tensor, response_mask, "first_k", first_k=first_k
    )
    assert compact_repr.shape == (1, first_k, 8)
    assert int(compact_mask.sum().item()) == valid_len


def test_last_k_uses_configured_k_when_response_is_long_enough():
    response_len = 128
    last_k = 25
    response_mask = torch.ones(1, response_len, dtype=torch.float32)

    assert get_batch_distillation_k(response_mask, last_k) == last_k
    compact_repr, compact_mask = compact_response_repr_by_positions(
        torch.randn(1, response_len, 8), response_mask, "last_k", last_k=last_k
    )
    assert compact_repr.shape == (1, last_k, 8)
    assert int(compact_mask.sum().item()) == last_k


def test_mixed_valid_lengths_last_k_always_width_k_batch():
    response_len = 16384
    last_k = 4000
    response_mask = torch.zeros(2, response_len, dtype=torch.float32)
    response_mask[0, :3228] = 1.0
    response_mask[1, :4000] = 1.0
    repr_tensor = torch.randn(2, response_len, 8)

    compact_repr, compact_mask = compact_response_repr_by_positions(
        repr_tensor, response_mask, "last_k", last_k=last_k
    )
    assert compact_repr.shape == (2, last_k, 8)
    assert int(compact_mask[0].sum().item()) == 3228
    assert int(compact_mask[1].sum().item()) == 4000


def test_mixed_batch_last_k_pads_short_samples_to_batch_k():
    response_len = 128
    last_k = 25
    response_mask = torch.zeros(2, response_len, dtype=torch.float32)
    response_mask[0, :10] = 1.0
    response_mask[1, :] = 1.0

    compact_mask = build_compact_rep_distillation_position_mask(response_mask, "last_k", last_k=last_k)
    assert compact_mask.shape == (2, last_k)
    assert int(compact_mask[0].sum().item()) == 10
    assert int(compact_mask[1].sum().item()) == last_k


def test_proportional_layer_indices_map_student_depth_to_teacher():
    indices = get_proportional_layer_indices(28, 36)
    assert indices.shape == (28,)
    assert indices[0].item() == 0
    assert indices[-1].item() == 35
    assert indices[14].item() in (17, 18)


def test_align_teacher_layers_to_student_for_cross_arch_rep():
    student = torch.randn(2, 28, 100, 64)
    teacher = torch.randn(2, 36, 100, 128)
    aligned_student, aligned_teacher = align_teacher_layers_to_student(student, teacher)
    assert aligned_student.shape == (2, 28, 100, 64)
    assert aligned_teacher.shape == (2, 28, 100, 128)
    loss = multi_layer_normalized_mse_loss(aligned_student, aligned_teacher, num_layers=28)
    assert loss.ndim == 0


def test_multi_layer_mse_aligns_mismatched_depths():
    student = torch.randn(2, 28, 8)
    teacher = torch.randn(2, 36, 8)
    loss = multi_layer_normalized_mse_loss(student, teacher, num_layers=36)
    assert loss.ndim == 0
