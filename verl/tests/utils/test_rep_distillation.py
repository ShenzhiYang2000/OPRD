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
    LowRankCrossArchProjector,
    ResidualLowRankCrossArchProjector,
    build_student_subspace_projector,
    subtract_rowspace_projection,
    align_teacher_layers_to_student,
    compute_rep_alignment_metrics,
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


def test_compute_rep_alignment_metrics_returns_numeric_diagnostics():
    student = torch.randn(2, 28, 16, 64)
    teacher = torch.randn(2, 36, 16, 128)
    projector = LowRankCrossArchProjector(
        num_layers=28,
        student_dim=64,
        teacher_dim=128,
        rank=8,
        num_teacher_layers=36,
    )
    projector.maybe_init_teacher_pca_from_batch(teacher, num_layers=28)
    z_student, z_teacher = projector.project_pair(student, teacher, num_layers=28)
    param_metrics = projector.projector_param_metrics()
    assert param_metrics["rep/teacher_pt_initialized"] == 1.0
    assert param_metrics["rep/ps_weight_norm_mean"] > 0.0
    assert param_metrics["rep/pt_weight_norm_mean"] > 0.0

    metrics = compute_rep_alignment_metrics(z_student, z_teacher, num_layers=28)
    assert metrics["rep/subspace_dim"] == 8.0
    assert metrics["rep/num_aligned_layers"] == 28.0
    assert -1.0 <= metrics["rep/subspace_cosine"] <= 1.0
    assert metrics["rep/subspace_mse"] >= 0.0


def test_subtract_rowspace_projection_zeros_parallel_component():
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hidden = torch.tensor([[3.0, 4.0]])
    residual = subtract_rowspace_projection(hidden, weight)
    assert torch.allclose(residual, torch.zeros_like(hidden), atol=1e-6)


def test_residual_projector_projects_tail_after_head_removal():
    student = torch.randn(2, 4, 8, 32)
    teacher = torch.randn(2, 6, 8, 64)
    projector = ResidualLowRankCrossArchProjector(
        num_layers=4,
        student_dim=32,
        teacher_dim=64,
        head_rank=2,
        tail_rank=3,
        num_teacher_layers=6,
    )
    for layer_idx in range(4):
        q, _ = torch.linalg.qr(torch.randn(64, 2))
        projector.head_teacher_weights[layer_idx].copy_(q.T)
        projector.head_student_weights[layer_idx].copy_(torch.randn(2, 32))
    projector._head_loaded = True
    projector.maybe_init_tail_pca_from_batch(teacher, num_layers=4)
    z_s, z_t = projector.project_pair(student, teacher, num_layers=4)
    assert z_s.shape == (2, 4, 8, 3)
    assert z_t.shape == (2, 4, 8, 3)


def test_low_rank_projector_freeze_ps_excludes_student_projectors_from_trainable_params():
    projector = LowRankCrossArchProjector(
        num_layers=2,
        student_dim=64,
        teacher_dim=128,
        rank=8,
        num_teacher_layers=4,
    )
    assert len(projector.trainable_parameters()) == 2
    projector.freeze_student_projectors()
    assert projector.trainable_parameters() == []
    assert projector.projector_param_metrics()["rep/ps_frozen"] == 1.0


def test_low_rank_projector_maps_cross_arch_repr_to_shared_subspace():
    student = torch.randn(2, 28, 16, 64)
    teacher = torch.randn(2, 36, 16, 128)
    projector = LowRankCrossArchProjector(
        num_layers=28,
        student_dim=64,
        teacher_dim=128,
        rank=8,
        num_teacher_layers=36,
    )
    projector.maybe_init_teacher_pca_from_batch(teacher, num_layers=28)
    z_student, z_teacher = projector.project_pair(student, teacher, num_layers=28)
    assert z_student.shape == (2, 28, 16, 8)
    assert z_teacher.shape == (2, 28, 16, 8)
    loss = multi_layer_normalized_mse_loss(z_student, z_teacher, num_layers=28)
    assert loss.ndim == 0


def test_low_rank_projector_loads_mlp_preexp_checkpoint(tmp_path):
    student_dim = 64
    teacher_dim = 128
    rank = 8
    hidden_dim = 32
    num_layers = 2
    num_teacher_layers = 4

    state_dict = {}
    frozen_weights = {}
    frozen_means = {}
    for layer_idx in range(num_layers):
        key = f"s{layer_idx}_t{layer_idx}"
        mlp = build_student_subspace_projector(
            student_dim,
            rank,
            projector_type="mlp",
            mlp_hidden_mult=4,
            mlp_hidden_dim=hidden_dim,
        )
        state_dict[f"projectors.{key}.0.weight"] = mlp[0].weight.detach().clone()
        state_dict[f"projectors.{key}.2.weight"] = mlp[2].weight.detach().clone()
        frozen_weights[key] = torch.randn(rank, teacher_dim)
        frozen_means[key] = torch.randn(teacher_dim)

    ckpt_path = tmp_path / "ps_bank.pt"
    torch.save(
        {
            "rank": rank,
            "projector_type": "mlp",
            "mlp_hidden_mult": 4,
            "state_dict": state_dict,
            "frozen_pt_weights": frozen_weights,
            "frozen_pt_means": frozen_means,
        },
        ckpt_path,
    )

    projector = LowRankCrossArchProjector(
        num_layers=num_layers,
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        rank=rank,
        num_teacher_layers=num_teacher_layers,
        ps_projector_type="mlp",
        mlp_hidden_mult=4,
        mlp_hidden_dim=hidden_dim,
    )
    projector.load_from_preexp_checkpoint(ckpt_path)
    assert projector._loaded_ps_layers == num_layers
    assert projector._loaded_pt_layers == num_layers
    assert projector.ps_projector_type == "mlp"
    assert projector.projector_param_metrics()["rep/ps_projector_type_mlp"] == 1.0

    student = torch.randn(1, num_layers, 4, student_dim)
    teacher = torch.randn(1, num_teacher_layers, 4, teacher_dim)
    z_student, z_teacher = projector.project_pair(student, teacher, num_layers=num_layers)
    assert z_student.shape == (1, num_layers, 4, rank)
    assert z_teacher.shape == (1, num_layers, 4, rank)


def test_direct_low_rank_projector_loads_preexp_checkpoint(tmp_path):
    from verl.utils.rep_distillation import DirectLowRankCrossArchProjector, get_proportional_layer_indices

    student_dim = 64
    teacher_dim = 128
    rank = 8
    num_layers = 2
    num_teacher_layers = 4
    teacher_layer_indices = get_proportional_layer_indices(num_layers, num_teacher_layers).tolist()

    state_dict = {}
    pt_state_dict = {}
    for layer_idx, teacher_layer in enumerate(teacher_layer_indices):
        key = f"s{layer_idx}_t{teacher_layer}"
        ps = build_student_subspace_projector(student_dim, rank, projector_type="linear")
        pt = build_student_subspace_projector(teacher_dim, rank, projector_type="linear")
        state_dict[f"projectors.{key}.weight"] = ps.weight.detach().clone()
        pt_state_dict[f"projectors.{key}.weight"] = pt.weight.detach().clone()

    ckpt_path = tmp_path / "direct_bank.pt"
    torch.save(
        {
            "subspace_mode": "direct",
            "rank": rank,
            "projector_type": "linear",
            "state_dict": state_dict,
            "pt_state_dict": pt_state_dict,
        },
        ckpt_path,
    )

    projector = DirectLowRankCrossArchProjector(
        num_layers=num_layers,
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        rank=rank,
        num_teacher_layers=num_teacher_layers,
    )
    projector.load_from_preexp_checkpoint(ckpt_path)
    assert projector._loaded_ps_layers == num_layers
    assert projector._loaded_pt_layers == num_layers
    assert projector.projector_param_metrics()["rep/subspace_mode_direct"] == 1.0

    student = torch.randn(1, num_layers, 4, student_dim)
    teacher = torch.randn(1, num_teacher_layers, 4, teacher_dim)
    z_student, z_teacher = projector.project_pair(student, teacher, num_layers=num_layers)
    assert z_student.shape == (1, num_layers, 4, rank)
    assert z_teacher.shape == (1, num_layers, 4, rank)
