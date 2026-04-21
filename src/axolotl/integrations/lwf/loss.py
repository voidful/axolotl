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
Loss helpers for Learning without Forgetting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from axolotl.integrations.forgetting_common import compute_next_token_ce_loss


def compute_lwf_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ce_alpha: float = 1.0,
    alpha: float = 1.0,
    temperature: float = 2.0,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combine token CE with a teacher KL term on the same inputs.
    """
    ce_loss, ce_stats = compute_next_token_ce_loss(
        logits=student_logits,
        labels=labels,
        num_items_in_batch=num_items_in_batch,
    )

    student_shift = student_logits[..., :-1, :].contiguous()
    teacher_shift = teacher_logits[..., :-1, :].contiguous()
    shift_mask = ce_stats["shift_mask"]
    mask_float = shift_mask.float()

    student_log_probs = F.log_softmax(student_shift / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_shift / temperature, dim=-1)
    token_kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)

    if num_items_in_batch is not None:
        denom = student_logits.new_tensor(max(float(num_items_in_batch), 1.0))
    else:
        denom = mask_float.sum().clamp(min=1.0)

    kd_loss = (token_kl * mask_float).sum() / denom
    kd_loss = kd_loss * (temperature ** 2)

    total_loss = (ce_alpha * ce_loss) + (alpha * kd_loss)
    return total_loss, {
        "ce_loss": ce_loss.detach(),
        "kd_loss": kd_loss.detach(),
        "shift_mask": shift_mask,
        "token_kl": token_kl.detach(),
    }
