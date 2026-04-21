# Copyright 2024 Axolotl AI. All rights reserved.
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
Shared utilities for forgetting-aware regularization baselines.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def canonical_param_name(name: str) -> str:
    """
    Normalize wrapper-specific prefixes in parameter names.
    """
    for prefix in ("module.", "_orig_mod."):
        if name.startswith(prefix):
            return canonical_param_name(name[len(prefix):])
    return name


def iter_named_trainable_params(
    model,
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """
    Yield canonical parameter names for trainable parameters only.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield canonical_param_name(name), param


def snapshot_trainable_params(model) -> dict[str, torch.Tensor]:
    """
    Clone the current trainable parameters as detached references.
    """
    return {
        name: param.detach().clone()
        for name, param in iter_named_trainable_params(model)
    }


def compute_next_token_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Standard next-token cross-entropy with ignore_index support.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = shift_labels != -100

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )

    if num_items_in_batch is not None:
        denom = logits.new_tensor(max(float(num_items_in_batch), 1.0))
    else:
        denom = shift_mask.sum().float().clamp(min=1.0)

    return loss / denom, {
        "shift_logits": shift_logits,
        "shift_labels": shift_labels,
        "shift_mask": shift_mask,
        "denom": denom,
    }


def quadratic_reference_penalty(
    named_params: dict[str, torch.Tensor],
    reference_params: dict[str, torch.Tensor],
    fisher_diagonal: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Compute a quadratic parameter anchoring penalty.

    If `fisher_diagonal` is provided, the penalty becomes a diagonal EWC penalty.
    """
    if not named_params:
        raise ValueError("named_params must not be empty")

    first_param = next(iter(named_params.values()))
    penalty = first_param.new_zeros(())

    for name, param in named_params.items():
        ref = reference_params.get(name)
        if ref is None:
            continue
        diff = param - ref.to(device=param.device, dtype=param.dtype)
        if fisher_diagonal is None:
            penalty = penalty + diff.pow(2).sum()
        else:
            fisher = fisher_diagonal.get(name)
            if fisher is None:
                continue
            penalty = penalty + (
                fisher.to(device=param.device, dtype=param.dtype) * diff.pow(2)
            ).sum()

    return 0.5 * penalty
