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
Core loss function for the token-level focal-loss baseline.
"""

from __future__ import annotations

import torch


def compute_focal_loss(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-6,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute token-level focal loss for next-token prediction.

    Args:
        active_logits: Logits from active model, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T), with -100 ignored.
        gamma: Focal sharpening parameter. gamma=0 recovers CE.
        eps: Numerical stability term.
        num_items_in_batch: Optional packed-sequence reduction denominator.

    Returns:
        Tuple of scalar loss and intermediate tensors for logging.
    """
    if gamma < 0:
        raise ValueError(f"focal gamma must be >= 0, got {gamma}")

    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_mask = shift_labels != -100
    mask_float = shift_mask.float()
    safe_labels = shift_labels.clamp(min=0)

    target_logit = shift_logits.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    log_z = torch.logsumexp(shift_logits, dim=-1)
    target_logp = (target_logit - log_z) * mask_float
    ce_t = -target_logp
    pt = target_logp.exp() * mask_float

    if gamma == 0.0:
        focal_weight = mask_float.clone()
    else:
        focal_weight = (1.0 - pt).clamp(min=eps).pow(gamma) * mask_float

    weighted_ce = focal_weight * ce_t * mask_float

    if num_items_in_batch is not None:
        denom = active_logits.new_tensor(max(float(num_items_in_batch), 1.0))
    else:
        denom = mask_float.sum().clamp(min=1.0)

    loss = weighted_ce.sum() / denom

    return loss, {
        "shift_mask": shift_mask,
        "target_logp": target_logp.detach(),
        "ce_t": ce_t.detach(),
        "pt": pt.detach(),
        "focal_weight": focal_weight.detach(),
    }
