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
Core loss for entropy-focus baselines.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def compute_entropy_focus_loss(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    mode: Literal["high", "low"] = "high",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Keep only high-entropy or low-entropy valid tokens within a batch.

    Tokens are split by the mean predictive entropy over valid positions.
    """
    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = shift_labels != -100
    mask_float = shift_mask.float()
    safe_labels = shift_labels.clamp(min=0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = log_probs.exp()

    entropy_t = -(probs * log_probs).sum(dim=-1) * mask_float
    target_logp = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1) * mask_float
    ce_t = -target_logp

    if shift_mask.any():
        mean_entropy = entropy_t[shift_mask].detach().mean()
    else:
        mean_entropy = entropy_t.new_zeros(())

    if mode == "high":
        selected_mask = shift_mask & (entropy_t >= (mean_entropy - eps))
    elif mode == "low":
        selected_mask = shift_mask & (entropy_t < mean_entropy)
    else:
        raise ValueError(f"Unsupported entropy focus mode {mode!r}")

    if not selected_mask.any():
        selected_mask = shift_mask.clone()

    selected_float = selected_mask.float()
    denom = selected_float.sum().clamp(min=1.0)
    loss = (ce_t * selected_float).sum() / denom

    return loss, {
        "shift_mask": shift_mask,
        "selected_mask": selected_mask,
        "entropy_t": entropy_t.detach(),
        "mean_entropy": mean_entropy.detach(),
        "ce_t": ce_t.detach(),
    }
