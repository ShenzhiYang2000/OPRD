# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.att_distillation import (
    att_distillation_query_valid_fraction,
    attention_distillation_loss,
    disable_gradient_checkpointing,
    forward_response_attn_rows,
    get_att_distillation_context_metadata,
    validate_att_distillation_loss_type,
)
from verl.utils.rep_distillation import (
    LowRankCrossArchProjector,
    DirectLowRankCrossArchProjector,
    ResidualLowRankCrossArchProjector,
    infer_subspace_mode_from_checkpoint,
    load_preexp_checkpoint_dict,
    resolve_student_projector_settings,
    _squeeze_hidden_rmpad,
    build_compact_rep_distillation_position_mask,
    build_rep_distillation_position_mask,
    compact_response_repr_by_positions,
    extract_teacher_response_hidden_repr,
    forward_response_hidden_repr,
    hidden_states_tuple_to_response_repr,
    compute_rep_alignment_metrics,
    multi_layer_normalized_cosine_similarity,
    multi_layer_normalized_mse_loss,
    validate_rep_distillation_layers,
    validate_rep_distillation_positions,
    validate_rep_projector_mode,
)
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.rep_projector = None
        self._rep_projector_in_optimizer = False
        self.low_rank_projector: LowRankCrossArchProjector | DirectLowRankCrossArchProjector | None = None
        self._low_rank_projector_in_optimizer = False
        self.residual_low_rank_projector: ResidualLowRankCrossArchProjector | None = None
        self._residual_low_rank_projector_in_optimizer = False

    def _get_or_create_rep_projector(self, student_dim: int, teacher_dim: int) -> nn.Linear | None:
        if student_dim == teacher_dim:
            return None
        if self.rep_projector is None:
            self.rep_projector = nn.Linear(student_dim, teacher_dim, bias=False).to(get_device_id())
            if self.actor_optimizer is not None and not self._rep_projector_in_optimizer:
                self.actor_optimizer.add_param_group({"params": self.rep_projector.parameters()})
                self._rep_projector_in_optimizer = True
        return self.rep_projector

    def _get_or_create_low_rank_projector(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        num_layers: int,
        num_teacher_layers: int,
    ) -> LowRankCrossArchProjector | DirectLowRankCrossArchProjector:
        if self.low_rank_projector is None:
            rank = int(self.config.get("rep_low_rank", 256))
            checkpoint = self.config.get("rep_low_rank_init_checkpoint", None)
            ps_type, mlp_mult, mlp_hidden_dim = resolve_student_projector_settings(
                ps_projector=str(self.config.get("rep_ps_projector", "auto")),
                mlp_hidden_mult=int(self.config.get("rep_mlp_hidden_mult", 4)),
                checkpoint_path=checkpoint,
            )
            subspace_mode = "full"
            if checkpoint:
                subspace_mode = infer_subspace_mode_from_checkpoint(
                    load_preexp_checkpoint_dict(checkpoint)
                )
            if subspace_mode == "direct":
                if ps_type != "linear":
                    raise ValueError(
                        f"Direct preexp2 checkpoint at {checkpoint} requires linear P_S, got {ps_type!r}"
                    )
                self.low_rank_projector = DirectLowRankCrossArchProjector(
                    num_layers=num_layers,
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    rank=rank,
                    num_teacher_layers=num_teacher_layers,
                ).to(get_device_id())
            else:
                self.low_rank_projector = LowRankCrossArchProjector(
                    num_layers=num_layers,
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    rank=rank,
                    num_teacher_layers=num_teacher_layers,
                    ps_projector_type=ps_type,
                    mlp_hidden_mult=mlp_mult,
                    mlp_hidden_dim=mlp_hidden_dim,
                ).to(get_device_id())
            if checkpoint:
                self.low_rank_projector.load_from_preexp_checkpoint(checkpoint)
            freeze_ps = bool(self.config.get("rep_freeze_ps", False))
            if freeze_ps:
                self.low_rank_projector.freeze_student_projectors()
                if self.low_rank_projector._loaded_ps_layers == 0:
                    print(
                        "WARNING: rep_freeze_ps=True but no P_S weights were loaded from checkpoint. "
                        "Rep loss will use random frozen P_S; set rep_low_rank_init_checkpoint."
                    )
            elif self.actor_optimizer is not None and not self._low_rank_projector_in_optimizer:
                trainable_params = self.low_rank_projector.trainable_parameters()
                if trainable_params:
                    self.actor_optimizer.add_param_group({"params": trainable_params})
                    self._low_rank_projector_in_optimizer = True
        return self.low_rank_projector

    def _get_or_create_residual_low_rank_projector(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        num_layers: int,
        num_teacher_layers: int,
    ) -> ResidualLowRankCrossArchProjector:
        if self.residual_low_rank_projector is None:
            head_rank = int(self.config.get("rep_head_rank", 16))
            tail_rank = int(self.config.get("rep_low_rank", 16))
            head_checkpoint = self.config.get("rep_head_init_checkpoint", None)
            tail_checkpoint = self.config.get("rep_low_rank_init_checkpoint", None)
            if not head_checkpoint:
                raise ValueError(
                    "rep_projector_mode=low_rank_residual requires rep_head_init_checkpoint "
                    "(pre-experiment 2 head ps_bank.pt, e.g. rank_16/ps_bank.pt)"
                )
            tail_ps_type, tail_mlp_mult, tail_mlp_hidden_dim = resolve_student_projector_settings(
                ps_projector=str(self.config.get("rep_ps_projector", "auto")),
                mlp_hidden_mult=int(self.config.get("rep_mlp_hidden_mult", 4)),
                checkpoint_path=tail_checkpoint,
            )
            self.residual_low_rank_projector = ResidualLowRankCrossArchProjector(
                num_layers=num_layers,
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                head_rank=head_rank,
                tail_rank=tail_rank,
                num_teacher_layers=num_teacher_layers,
                tail_ps_projector_type=tail_ps_type,
                tail_mlp_hidden_mult=tail_mlp_mult,
                tail_mlp_hidden_dim=tail_mlp_hidden_dim,
            ).to(get_device_id())
            self.residual_low_rank_projector.load_head_from_preexp_checkpoint(head_checkpoint)
            if tail_checkpoint:
                self.residual_low_rank_projector.load_tail_from_preexp_checkpoint(tail_checkpoint)
            freeze_ps = bool(self.config.get("rep_freeze_ps", False))
            if freeze_ps:
                self.residual_low_rank_projector.freeze_student_projectors()
                if self.residual_low_rank_projector._loaded_tail_ps_layers == 0:
                    print(
                        "WARNING: rep_freeze_ps=True but no tail P_S loaded. "
                        "Set rep_low_rank_init_checkpoint to a residual ps_tail_bank.pt."
                    )
            elif self.actor_optimizer is not None and not self._residual_low_rank_projector_in_optimizer:
                trainable_params = self.residual_low_rank_projector.trainable_parameters()
                if trainable_params:
                    self.actor_optimizer.add_param_group({"params": trainable_params})
                    self._residual_low_rank_projector_in_optimizer = True
        return self.residual_low_rank_projector

    def _extract_response_hidden_repr_from_output(
        self,
        output,
        response_mask: torch.Tensor,
        *,
        indices=None,
        batch_size: int | None = None,
        seqlen: int | None = None,
        pad_size: int = 0,
        rep_distillation_positions: str = "all",
        rep_distillation_layers: str = "last",
        rep_distillation_last_k: int = 32,
        rep_distillation_first_k: int = 50,
    ) -> torch.Tensor:
        if rep_distillation_layers != "last":
            if self.use_remove_padding:
                return hidden_states_tuple_to_response_repr(
                    output.hidden_states,
                    response_mask,
                    rep_distillation_positions,
                    rep_distillation_layers,
                    indices=indices,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    use_ulysses_sp=self.use_ulysses_sp,
                    pad_size=pad_size,
                    use_remove_padding=True,
                    last_k=rep_distillation_last_k,
                    first_k=rep_distillation_first_k,
                )
            return extract_teacher_response_hidden_repr(
                output.hidden_states,
                response_mask,
                rep_distillation_positions,
                rep_distillation_layers,
                last_k=rep_distillation_last_k,
                first_k=rep_distillation_first_k,
            )

        if self.use_remove_padding:
            last_hidden_rmpad = _squeeze_hidden_rmpad(output.hidden_states[-1])
            if self.use_ulysses_sp:
                last_hidden_rmpad = gather_outputs_and_unpad(
                    last_hidden_rmpad,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
            full_last_hidden = pad_input(
                hidden_states=last_hidden_rmpad,
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
            hidden_states = full_last_hidden.float()
        else:
            hidden_states = output.hidden_states[-1].float()

        return extract_teacher_response_hidden_repr(
            hidden_states,
            response_mask,
            rep_distillation_positions,
            "last",
            last_k=rep_distillation_last_k,
            first_k=rep_distillation_first_k,
        )

    def _forward_response_hidden_repr(self, micro_batch) -> torch.Tensor:
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        rep_distillation_positions = validate_rep_distillation_positions(
            self.config.get("rep_distillation_positions", "last")
        )
        rep_distillation_layers = validate_rep_distillation_layers(
            self.config.get("rep_distillation_layers", "last")
        )
        rep_distillation_last_k = int(self.config.get("rep_distillation_last_k", 32))
        rep_distillation_first_k = int(self.config.get("rep_distillation_first_k", 50))

        return forward_response_hidden_repr(
            self.actor_module,
            micro_batch["input_ids"],
            micro_batch["attention_mask"],
            micro_batch["position_ids"],
            micro_batch["response_mask"],
            use_remove_padding=self.use_remove_padding,
            use_ulysses_sp=self.use_ulysses_sp,
            ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
            multi_modal_inputs=multi_modal_inputs,
            positions=rep_distillation_positions,
            layers=rep_distillation_layers,
            last_k=rep_distillation_last_k,
            first_k=rep_distillation_first_k,
        )

    def _forward_response_attn_rows(self, micro_batch) -> torch.Tensor:
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        att_distillation_positions = validate_rep_distillation_positions(
            self.config.get("att_distillation_positions", "last")
        )
        att_distillation_layers = validate_rep_distillation_layers(
            self.config.get("att_distillation_layers", "last")
        )

        return forward_response_attn_rows(
            self.actor_module,
            micro_batch["input_ids"],
            micro_batch["attention_mask"],
            micro_batch["position_ids"],
            micro_batch["response_mask"],
            multi_modal_inputs=multi_modal_inputs,
            positions=att_distillation_positions,
            layers=att_distillation_layers,
            last_k=int(self.config.get("att_distillation_last_k", 32)),
            first_k=int(self.config.get("att_distillation_first_k", 50)),
            max_context_len=int(self.config.get("att_distillation_max_key_len", 4096)),
        )

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k=0,
        student_top_k_ids=None,
        return_response_hidden_repr: bool = False,
        rep_distillation_positions: str = "all",
        rep_distillation_layers: str = "last",
        rep_distillation_last_k: int = 32,
        rep_distillation_first_k: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
            response_hidden_repr: # (bs, response_len, hidden_dim) or None
        """
        response_length = micro_batch["responses"].size(-1)
        response_mask = None
        if return_response_hidden_repr:
            if "response_mask" in micro_batch:
                response_mask = micro_batch["response_mask"]
            else:
                response_mask = micro_batch["attention_mask"][:, -response_length:]
        response_hidden_repr = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                pad_size = 0
                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_hidden_repr,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_response_hidden_repr:
                    response_hidden_repr = self._extract_response_hidden_repr_from_output(
                        output,
                        response_mask,
                        indices=indices,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        pad_size=pad_size,
                        rep_distillation_positions=rep_distillation_positions,
                        rep_distillation_layers=rep_distillation_layers,
                        rep_distillation_last_k=rep_distillation_last_k,
                        rep_distillation_first_k=rep_distillation_first_k,
                    )

                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_hidden_repr,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_response_hidden_repr:
                    response_hidden_repr = self._extract_response_hidden_repr_from_output(
                        output,
                        response_mask,
                        rep_distillation_positions=rep_distillation_positions,
                        rep_distillation_layers=rep_distillation_layers,
                        rep_distillation_last_k=rep_distillation_last_k,
                        rep_distillation_first_k=rep_distillation_first_k,
                    )

                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs, response_hidden_repr

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
            
        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_rep_distillation = self.config.get("use_rep_distillation", False)
        use_att_distillation = self.config.get("use_att_distillation", False)
        rep_distillation_only = self.config.get("rep_distillation_only", False)
        rep_distillation_coef = self.config.get("rep_distillation_coef", 1.0)
        att_distillation_coef = self.config.get("att_distillation_coef", 1.0)
        att_distillation_loss_type = validate_att_distillation_loss_type(
            self.config.get("att_distillation_loss", "kl")
        )
        att_distillation_temperature = float(self.config.get("att_distillation_temperature", 1.0))
        att_distillation_positions = validate_rep_distillation_positions(
            self.config.get("att_distillation_positions", "last")
        )
        att_distillation_layers = validate_rep_distillation_layers(
            self.config.get("att_distillation_layers", "last")
        )
        att_distillation_last_k = int(self.config.get("att_distillation_last_k", 32))
        att_distillation_first_k = int(self.config.get("att_distillation_first_k", 50))
        multi_layer_att = att_distillation_layers != "last"
        rep_distillation_positions = validate_rep_distillation_positions(
            self.config.get("rep_distillation_positions", "last")
        )
        rep_distillation_layers = validate_rep_distillation_layers(
            self.config.get("rep_distillation_layers", "last")
        )
        rep_distillation_last_k = int(self.config.get("rep_distillation_last_k", 32))
        rep_distillation_first_k = int(self.config.get("rep_distillation_first_k", 50))
        multi_layer_rep = rep_distillation_layers != "last"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
        if not rep_distillation_only:
            select_keys.extend(["old_log_probs", "advantages"])
        if use_rep_distillation:
            select_keys.append("teacher_last_hidden_repr")
        if use_att_distillation:
            select_keys.append("teacher_attn_rows")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    # PG (varlen) + att (eager padded) share one backward; disable activation
                    # checkpointing to avoid FSDP/shape conflicts on the second forward.
                    grad_ckpt_ctx = (
                        disable_gradient_checkpointing(self.actor_module)
                        if use_att_distillation
                        else nullcontext()
                    )
                    old_log_prob = model_inputs.get("old_log_probs")
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    policy_loss = None
                    student_repr_from_forward = None
                    att_loss = None
                    with grad_ckpt_ctx:
                        if not rep_distillation_only:
                            # all return: (bsz, response_length)
                            calculate_entropy = False
                            if entropy_coeff != 0:
                                calculate_entropy = True

                            # Reuse the same forward for rep loss when combined with OPD (avoids a
                            # second full forward that breaks activation offload).
                            return_hidden = use_rep_distillation

                            # Check if we have 3D advantages (top-k sampling case)
                            # If so, we need to recompute top-k log probs for correct gradient
                            if advantages.dim() == 3:
                                top_k = advantages.shape[-1]
                                # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                                student_top_k_ids = None
                                if "union_top_k_ids" in model_inputs:
                                    student_top_k_ids = model_inputs["union_top_k_ids"]
                                elif "student_top_k_ids" in model_inputs:
                                    student_top_k_ids = model_inputs["student_top_k_ids"]

                                entropy, _, _, topk_log_probs, student_repr_from_forward = self._forward_micro_batch(
                                    model_inputs,
                                    temperature=temperature,
                                    calculate_entropy=calculate_entropy,
                                    top_k=top_k,
                                    student_top_k_ids=student_top_k_ids,
                                    return_response_hidden_repr=return_hidden,
                                    rep_distillation_positions=rep_distillation_positions,
                                    rep_distillation_layers=rep_distillation_layers,
                                    rep_distillation_last_k=rep_distillation_last_k,
                                    rep_distillation_first_k=rep_distillation_first_k,
                                )
                                log_prob_for_loss = topk_log_probs

                            else:
                                _, log_prob, _, _, student_repr_from_forward = self._forward_micro_batch(
                                    model_inputs,
                                    temperature=temperature,
                                    calculate_entropy=calculate_entropy,
                                    return_response_hidden_repr=return_hidden,
                                    rep_distillation_positions=rep_distillation_positions,
                                    rep_distillation_layers=rep_distillation_layers,
                                    rep_distillation_last_k=rep_distillation_last_k,
                                    rep_distillation_first_k=rep_distillation_first_k,
                                )
                                log_prob_for_loss = log_prob

                            format_mask = None
                            if "format_mask" in model_inputs.keys():
                                format_mask = model_inputs["format_mask"]

                            # for fully_async_policy recipe
                            if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                                old_log_prob = model_inputs["old_log_probs"]
                            else:
                                if on_policy:
                                    print("on_policy")
                                    # For on-policy (ppo_epochs=1), use current policy as "old"
                                    # log_prob_for_loss is already 3D for top-k case
                                    old_log_prob = log_prob_for_loss.detach()
                                else:
                                    print("off_policy")
                                    # For off-policy, use stored log probs
                                    # For 3D top-k case, use stored log probs (union or student)
                                    if advantages.dim() == 3:
                                        if "union_top_k_log_probs" in model_inputs:
                                            old_log_prob = model_inputs["union_top_k_log_probs"]
                                        elif "student_top_k_log_probs" in model_inputs:
                                            old_log_prob = model_inputs["student_top_k_log_probs"]
                                        else:
                                            old_log_prob = model_inputs["old_log_probs"]
                                    else:
                                        old_log_prob = model_inputs["old_log_probs"]

                            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                            # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                            # Extract pre-computed rollout correction weights if present
                            # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                            rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                            # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                            # are computed centrally in ray_trainer.py for consistency and efficiency.
                            # This ensures metrics are computed uniformly across all batches at the trainer level
                            # and avoids redundant computation across workers and micro-batches.

                            # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                            # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                            policy_loss_fn = get_policy_loss_fn(loss_mode)

                            # Compute policy loss (any function is expected to return 2 values)
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                                format_mask=format_mask,
                            )
                            micro_batch_metrics.update(pg_metrics)

                            if entropy_coeff != 0:
                                entropy_loss = agg_loss(
                                    loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                                )

                                # compute policy loss
                                policy_loss = pg_loss - entropy_loss * entropy_coeff
                            else:
                                policy_loss = pg_loss

                            if self.config.use_kl_loss:
                                ref_log_prob = model_inputs["ref_log_prob"]
                                # compute kl loss
                                kld = kl_penalty(
                                    logprob=log_prob_for_loss,
                                    ref_logprob=ref_log_prob,
                                    kl_penalty=self.config.kl_loss_type,
                                )
                                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                                micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                            micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor

                        if use_rep_distillation:
                            teacher_repr = model_inputs["teacher_last_hidden_repr"]
                            response_mask = model_inputs["response_mask"]
                            if student_repr_from_forward is not None:
                                student_repr = student_repr_from_forward
                            else:
                                student_repr = self._forward_response_hidden_repr(model_inputs)

                            num_rep_layers = student_repr.size(1) if multi_layer_rep else None
                            position_mask = None

                            if rep_distillation_positions == "last":
                                if multi_layer_rep:
                                    if student_repr.dim() != 3:
                                        raise ValueError(
                                            "Expected student multi-layer last-token repr (B, L, D)"
                                        )
                                else:
                                    if student_repr.dim() != 3:
                                        raise ValueError(
                                            "Expected student response hidden states (B, T, D) for last-token rep distillation"
                                        )
                                    last_indices = response_mask.long().sum(dim=1) - 1
                                    last_indices = last_indices.clamp(min=0)
                                    batch_indices = torch.arange(student_repr.size(0), device=student_repr.device)
                                    student_repr = student_repr[batch_indices, last_indices]
                            elif rep_distillation_positions in ("first_k", "last_k"):
                                if student_repr.shape != teacher_repr.shape:
                                    student_repr, _ = compact_response_repr_by_positions(
                                        student_repr,
                                        response_mask,
                                        rep_distillation_positions,
                                        rep_distillation_last_k,
                                        rep_distillation_first_k,
                                    )
                                position_mask = build_compact_rep_distillation_position_mask(
                                    response_mask,
                                    rep_distillation_positions,
                                    rep_distillation_last_k,
                                    rep_distillation_first_k,
                                )
                                if multi_layer_rep:
                                    if student_repr.dim() != 4 or teacher_repr.dim() != 4:
                                        raise ValueError(
                                            "Expected compact multi-layer repr (B, L, k, D) for first_k/last_k"
                                        )
                                elif student_repr.dim() != 3 or teacher_repr.dim() != 3:
                                    raise ValueError(
                                        "Expected compact single-layer repr (B, k, D) for first_k/last_k"
                                    )
                            else:
                                position_mask = response_mask
                                if multi_layer_rep:
                                    if student_repr.dim() != 4:
                                        raise ValueError(
                                            "Expected student multi-layer per-token repr (B, L, T, D)"
                                        )
                                elif student_repr.dim() != 3:
                                    raise ValueError(
                                        "teacher_last_hidden_repr is per-token but student repr is not (B, T, D)"
                                    )

                            student_dim = int(student_repr.size(-1))
                            teacher_dim = int(teacher_repr.size(-1))
                            micro_batch_metrics["rep/student_hidden_dim"] = float(student_dim)
                            micro_batch_metrics["rep/teacher_hidden_dim"] = float(teacher_dim)
                            if multi_layer_rep:
                                micro_batch_metrics["rep/num_student_layers"] = float(student_repr.size(1))
                                micro_batch_metrics["rep/num_teacher_layers_raw"] = float(teacher_repr.size(1))
                            rep_projector_mode = validate_rep_projector_mode(
                                self.config.get("rep_projector_mode", "full")
                            )
                            if student_dim == teacher_dim:
                                rep_projector_mode = "full"

                            if rep_projector_mode == "low_rank":
                                num_teacher_layers = teacher_repr.size(1) if multi_layer_rep else 1
                                low_rank_layers = num_rep_layers or 1
                                low_rank_projector = self._get_or_create_low_rank_projector(
                                    student_dim,
                                    teacher_dim,
                                    num_layers=low_rank_layers,
                                    num_teacher_layers=num_teacher_layers,
                                )
                                low_rank_projector.maybe_init_teacher_pca_from_batch(
                                    teacher_repr, num_layers=low_rank_layers
                                )
                                student_repr, teacher_repr = low_rank_projector.project_pair(
                                    student_repr,
                                    teacher_repr,
                                    num_layers=num_rep_layers,
                                )
                                micro_batch_metrics["rep/use_low_rank_projector"] = 1.0
                                micro_batch_metrics["rep/low_rank"] = float(low_rank_projector.rank)
                                micro_batch_metrics.update(low_rank_projector.projector_param_metrics())
                            elif rep_projector_mode == "low_rank_residual":
                                num_teacher_layers = teacher_repr.size(1) if multi_layer_rep else 1
                                low_rank_layers = num_rep_layers or 1
                                residual_projector = self._get_or_create_residual_low_rank_projector(
                                    student_dim,
                                    teacher_dim,
                                    num_layers=low_rank_layers,
                                    num_teacher_layers=num_teacher_layers,
                                )
                                micro_batch_metrics.update(
                                    residual_projector.head_alignment_metrics(
                                        student_repr,
                                        teacher_repr,
                                        num_layers=num_rep_layers,
                                    )
                                )
                                residual_projector.maybe_init_tail_pca_from_batch(
                                    teacher_repr, num_layers=low_rank_layers
                                )
                                student_repr, teacher_repr = residual_projector.project_pair(
                                    student_repr,
                                    teacher_repr,
                                    num_layers=num_rep_layers,
                                )
                                micro_batch_metrics["rep/use_low_rank_projector"] = 1.0
                                micro_batch_metrics["rep/low_rank"] = float(residual_projector.tail_rank)
                                micro_batch_metrics.update(residual_projector.projector_param_metrics())
                            else:
                                projector = self._get_or_create_rep_projector(student_dim, teacher_dim)
                                if projector is not None:
                                    student_repr = projector(student_repr)
                                    w = projector.weight.detach().float()
                                    micro_batch_metrics["rep/ps_weight_norm_mean"] = float(w.norm().item())
                                    micro_batch_metrics["rep/ps_weight_norm_max"] = float(w.norm().item())
                                    micro_batch_metrics["rep/full_projector_out_dim"] = float(w.shape[0])
                                    micro_batch_metrics["rep/full_projector_in_dim"] = float(w.shape[1])
                                micro_batch_metrics["rep/use_low_rank_projector"] = 0.0
                                micro_batch_metrics["rep/teacher_pt_initialized"] = 0.0
                                micro_batch_metrics["rep/projector_loaded_from_checkpoint"] = 0.0

                            rep_loss = multi_layer_normalized_mse_loss(
                                student_repr,
                                teacher_repr,
                                position_mask=position_mask,
                                loss_agg_mode=loss_agg_mode,
                                num_layers=num_rep_layers,
                            )
                            micro_batch_metrics.update(
                                compute_rep_alignment_metrics(
                                    student_repr,
                                    teacher_repr,
                                    position_mask=position_mask,
                                    num_layers=num_rep_layers,
                                )
                            )
                            # Backward-compatible aliases used in earlier logs.
                            micro_batch_metrics["rep/norm_mse"] = micro_batch_metrics["rep/subspace_mse"]
                            micro_batch_metrics["rep/cosine_similarity"] = micro_batch_metrics["rep/subspace_cosine"]
                            micro_batch_metrics["actor/rep_loss"] = rep_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/rep_weighted_loss"] = (
                                rep_loss.detach().item() * rep_distillation_coef * loss_scale_factor
                            )
                            micro_batch_metrics["actor/rep_distillation_coef"] = rep_distillation_coef
                            rep_term = rep_loss * rep_distillation_coef
                            if policy_loss is None:
                                policy_loss = rep_term
                            else:
                                policy_loss = policy_loss + rep_term

                        if use_att_distillation:
                            att_position_mask = None
                            if att_distillation_positions == "first_k":
                                att_position_mask = build_compact_rep_distillation_position_mask(
                                    response_mask,
                                    att_distillation_positions,
                                    att_distillation_last_k,
                                    att_distillation_first_k,
                                )
                            elif att_distillation_positions == "last_k":
                                att_position_mask = build_compact_rep_distillation_position_mask(
                                    response_mask,
                                    att_distillation_positions,
                                    att_distillation_last_k,
                                    att_distillation_first_k,
                                )
                            elif att_distillation_positions == "all":
                                att_position_mask = build_rep_distillation_position_mask(
                                    response_mask,
                                    att_distillation_positions,
                                    att_distillation_last_k,
                                    att_distillation_first_k,
                                )
                            original_seqlen, att_context_start, att_context_len = get_att_distillation_context_metadata(
                                model_inputs["input_ids"],
                                model_inputs["attention_mask"],
                                model_inputs["position_ids"],
                                response_mask,
                                positions=att_distillation_positions,
                                first_k=att_distillation_first_k,
                                last_k=att_distillation_last_k,
                                max_context_len=int(self.config.get("att_distillation_max_key_len", 4096)),
                            )
                            teacher_attn = model_inputs["teacher_attn_rows"]
                            num_att_layers = student_attn.size(1) if multi_layer_att else None
                            # Run att forward before backward, but merge into one backward pass.
                            # FSDP cannot safely run two sequential backward() calls on the same module.
                            student_attn = self._forward_response_attn_rows(model_inputs)
                            att_loss = attention_distillation_loss(
                                student_attn,
                                teacher_attn,
                                loss_type=att_distillation_loss_type,
                                position_mask=att_position_mask,
                                loss_agg_mode=loss_agg_mode,
                                temperature=att_distillation_temperature,
                                num_layers=num_att_layers,
                            )
                            micro_batch_metrics["att/valid_query_fraction"] = att_distillation_query_valid_fraction(
                                response_mask,
                                att_distillation_positions,
                                att_context_start,
                                att_context_len,
                                last_k=att_distillation_last_k,
                                first_k=att_distillation_first_k,
                                original_seqlen=original_seqlen,
                            )
                            micro_batch_metrics["att/context_len"] = float(att_context_len)
                            micro_batch_metrics["att/key_len_student"] = float(student_attn.shape[-1])
                            micro_batch_metrics["att/loss"] = att_loss.detach().item()
                            if num_att_layers is not None:
                                micro_batch_metrics["att/num_layers"] = num_att_layers
                            micro_batch_metrics["actor/att_loss"] = att_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/att_weighted_loss"] = (
                                att_loss.detach().item() * att_distillation_coef * loss_scale_factor
                            )
                            micro_batch_metrics["actor/att_distillation_coef"] = att_distillation_coef
                            del student_attn

                    if policy_loss is None and att_loss is None:
                        raise ValueError(
                            "No loss computed: enable use_rep_distillation, use_att_distillation, or policy gradient training."
                        )

                    total_loss = policy_loss
                    if att_loss is not None:
                        att_term = att_loss * att_distillation_coef
                        total_loss = att_term if total_loss is None else total_loss + att_term

                    (total_loss * loss_scale_factor).backward()

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
