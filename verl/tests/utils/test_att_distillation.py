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

from verl.utils.att_distillation import (
    _extract_rows_from_layer_attn,
    _get_absolute_response_query_indices,
    att_distillation_query_valid_fraction,
    extract_teacher_response_attn_rows,
    pad_attn_rows_to_key_width,
    slice_inputs_for_att_distillation,
)


def test_first_k_query_indices_use_original_seqlen():
    """Long responses must not shift response_start when context is truncated to first_k."""
    prompt_len = 145
    response_len = 11591
    first_k = 50
    original_seqlen = prompt_len + response_len

    input_ids = torch.zeros(1, original_seqlen, dtype=torch.long)
    attention_mask = torch.ones(1, original_seqlen, dtype=torch.long)
    position_ids = torch.arange(original_seqlen, dtype=torch.long).unsqueeze(0)
    response_mask = torch.ones(1, response_len, dtype=torch.float32)

    sliced_ids, _, _, context_start = slice_inputs_for_att_distillation(
        input_ids,
        attention_mask,
        position_ids,
        response_len,
        max_context_len=4096,
        positions="first_k",
        first_k=first_k,
    )
    context_len = sliced_ids.size(1)

    ctx_idx, valid = _get_absolute_response_query_indices(
        response_mask,
        "first_k",
        last_k=32,
        context_start=context_start,
        context_len=context_len,
        first_k=first_k,
        original_seqlen=original_seqlen,
    )

    assert context_start == 0
    assert context_len == prompt_len + first_k
    assert valid is not None
    assert ctx_idx.shape == (1, first_k)
    assert valid.shape == (1, first_k)
    assert ctx_idx[0, 0].item() == prompt_len
    assert ctx_idx[0, first_k - 1].item() == prompt_len + first_k - 1

    valid_fraction = att_distillation_query_valid_fraction(
        response_mask,
        "first_k",
        context_start,
        context_len,
        first_k=first_k,
        original_seqlen=original_seqlen,
    )
    assert valid_fraction == 1.0


def test_first_k_query_indices_batch_size():
    """Query indices must expand to the full micro-batch size."""
    batch_size = 4
    prompt_len = 145
    response_len = 11591
    first_k = 50
    original_seqlen = prompt_len + response_len

    response_mask = torch.ones(batch_size, response_len, dtype=torch.float32)
    sliced_len = prompt_len + first_k

    ctx_idx, valid = _get_absolute_response_query_indices(
        response_mask,
        "first_k",
        last_k=32,
        context_start=0,
        context_len=sliced_len,
        first_k=first_k,
        original_seqlen=original_seqlen,
    )

    assert ctx_idx.shape == (batch_size, first_k)
    assert valid.shape == (batch_size, first_k)
    assert valid.all()
    assert ctx_idx[0, 0].item() == ctx_idx[3, 0].item() == prompt_len


def test_first_k_extracted_rows_have_expected_key_len():
    prompt_len = 145
    response_len = 11591
    first_k = 50
    original_seqlen = prompt_len + response_len
    context_len = prompt_len + first_k
    num_heads = 4

    attn = torch.rand(1, num_heads, context_len, context_len)
    response_mask = torch.ones(1, response_len, dtype=torch.float32)

    rows = extract_teacher_response_attn_rows(
        (attn,),
        response_mask,
        "first_k",
        "last",
        context_start=0,
        context_len=context_len,
        first_k=first_k,
        original_seqlen=original_seqlen,
    )

    assert rows.shape == (1, first_k, num_heads, context_len)
    expected = _extract_rows_from_layer_attn(
        attn,
        torch.arange(prompt_len, prompt_len + first_k).unsqueeze(0),
        torch.ones(1, first_k),
    )
    assert torch.allclose(rows, expected)


def test_last_k_slice_uses_prompt_plus_last_k_on_long_response():
    """last_k context should be O(prompt + k), not the full max_key_len tail."""
    prompt_len = 145
    response_len = 11591
    last_k = 50
    original_seqlen = prompt_len + response_len

    input_ids = torch.zeros(1, original_seqlen, dtype=torch.long)
    attention_mask = torch.ones(1, original_seqlen, dtype=torch.long)
    position_ids = torch.arange(original_seqlen, dtype=torch.long).unsqueeze(0)
    response_mask = torch.ones(1, response_len, dtype=torch.float32)

    sliced_ids, _, _, context_start = slice_inputs_for_att_distillation(
        input_ids,
        attention_mask,
        position_ids,
        response_len,
        max_context_len=4096,
        positions="last_k",
        last_k=last_k,
        response_mask=response_mask,
    )
    context_len = sliced_ids.size(1)

    assert context_len == prompt_len + last_k
    assert context_start == original_seqlen - (prompt_len + last_k)

    ctx_idx, valid = _get_absolute_response_query_indices(
        response_mask,
        "last_k",
        last_k=last_k,
        context_start=context_start,
        context_len=context_len,
        original_seqlen=original_seqlen,
    )
    assert ctx_idx.shape == (1, last_k)
    assert valid.all()
    assert ctx_idx[0, -1].item() == original_seqlen - 1


def test_last_k_slice_ignores_left_padded_prompt_region():
    """Left-padded prompts must not inflate last_k context to response_start + k."""
    max_prompt_len = 2048
    actual_prompt_len = 145
    response_len = 16384
    last_k = 100
    valid_response_len = 10297
    original_seqlen = max_prompt_len + response_len

    input_ids = torch.zeros(1, original_seqlen, dtype=torch.long)
    attention_mask = torch.zeros(1, original_seqlen, dtype=torch.long)
    attention_mask[:, :actual_prompt_len] = 1
    attention_mask[:, max_prompt_len : max_prompt_len + valid_response_len] = 1
    position_ids = torch.arange(original_seqlen, dtype=torch.long).unsqueeze(0)
    response_mask = torch.zeros(1, response_len, dtype=torch.float32)
    response_mask[:, :valid_response_len] = 1.0

    sliced_ids, _, _, context_start = slice_inputs_for_att_distillation(
        input_ids,
        attention_mask,
        position_ids,
        response_len,
        max_context_len=4096,
        positions="last_k",
        last_k=last_k,
        response_mask=response_mask,
    )
    context_len = sliced_ids.size(1)

    assert context_len == actual_prompt_len + last_k
    assert context_start == max_prompt_len + valid_response_len - (actual_prompt_len + last_k)


def test_last_k_uses_all_tokens_when_response_shorter_than_k():
    prompt_len = 32
    response_len = 64
    last_k = 25
    valid_len = 10
    original_seqlen = prompt_len + response_len
    num_heads = 2

    input_ids = torch.zeros(1, original_seqlen, dtype=torch.long)
    attention_mask = torch.ones(1, original_seqlen, dtype=torch.long)
    position_ids = torch.arange(original_seqlen, dtype=torch.long).unsqueeze(0)
    response_mask = torch.zeros(1, response_len, dtype=torch.float32)
    response_mask[:, :valid_len] = 1.0

    sliced_ids, _, _, context_start = slice_inputs_for_att_distillation(
        input_ids,
        attention_mask,
        position_ids,
        response_len,
        max_context_len=4096,
        positions="last_k",
        last_k=last_k,
        response_mask=response_mask,
    )
    context_len = sliced_ids.size(1)
    assert context_len == prompt_len + valid_len

    attn = torch.rand(1, num_heads, context_len, context_len)
    rows = extract_teacher_response_attn_rows(
        (attn,),
        response_mask,
        "last_k",
        "last",
        context_start=context_start,
        context_len=context_len,
        last_k=last_k,
        original_seqlen=original_seqlen,
    )

    assert rows.shape == (1, last_k, num_heads, context_len)
    ctx_idx, valid = _get_absolute_response_query_indices(
        response_mask,
        "last_k",
        last_k=last_k,
        context_start=context_start,
        context_len=context_len,
        original_seqlen=original_seqlen,
    )
    assert ctx_idx.shape == (1, last_k)
    assert int(valid.sum().item()) == valid_len


def test_pad_attn_rows_to_key_width_right_aligns_causal_rows():
    rows = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]).unsqueeze(0)
    padded = pad_attn_rows_to_key_width(rows, 5)
    assert padded.shape == (1, 2, 5)
    assert torch.allclose(padded[0, 0], torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(padded[0, 1], torch.tensor([0.0, 0.0, 4.0, 5.0, 6.0]))
    assert pad_attn_rows_to_key_width(padded, 5) is padded
    truncated = pad_attn_rows_to_key_width(padded, 4)
    assert torch.allclose(truncated[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
