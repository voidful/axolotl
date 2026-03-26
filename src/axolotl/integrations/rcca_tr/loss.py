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
Core loss function for RCCA-TR: Self-Paced Hardness Suppression.

Only the active model's cross-entropy determines token weights.
No frozen prior cache is needed — the method is fully self-contained.

Single gate:
  h_t = σ((CE_t − τ_p) / T_p)

Weight:
  w_t = w_min + (1 − w_min) · (1 − h_t)

  - Easy token (CE low):  h_t ≈ 0  → w_t ≈ 1      (learn normally)
  - Hard token (CE high): h_t ≈ 1  → w_t ≈ w_min   (suppress)

Final loss:
  L = Σ w_t · CE_t / |V|
"""

import torch
import torch.nn.functional as F


def compute_hardness_gate(
    ce_t: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_p: float = 2.0,
    T_p: float = 1.0,
) -> torch.Tensor:
    """
    Hardness gate: is this token hard enough to warrant suppression?

    h_t = σ((CE_t − τ_p) / T_p)

    High h_t → token is hard, should be suppressed.
    Low h_t  → token is easy, learn normally.

    Args:
        ce_t: Per-token cross-entropy, shape (B, T).
        valid_mask: Boolean mask for valid tokens, shape (B, T).
        tau_p: Hardness threshold.
        T_p: Temperature for sigmoid smoothing.

    Returns:
        h_t: Hardness scores in [0, 1], shape (B, T).
    """
    h_t = torch.sigmoid((ce_t - tau_p) / T_p)
    return h_t * valid_mask.float()


def compute_weighted_ce_loss(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    tau_p: float = 2.0,
    T_p: float = 1.0,
    w_min: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the RCCA-TR self-paced weighted CE loss.

    L = Σ w_t · CE_t / |V|

    w_t = w_min + (1 − w_min) · (1 − h_t)

    All shifting for next-token prediction is handled internally.

    Args:
        active_logits: Logits from active model, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T). -100 = ignore.
        tau_p: Hardness threshold.
        T_p: Hardness temperature.
        w_min: Weight floor for hard tokens.

    Returns:
        Tuple of (scalar loss, dict of intermediate tensors for logging).
    """
    # Shift for next-token prediction
    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_mask = shift_labels != -100
    shift_safe_labels = shift_labels.clamp(min=0)

    # Per-token CE
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logp = log_probs.gather(
        dim=-1, index=shift_safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    ce_t = -per_token_logp  # (B, T-1)

    # --- Hardness gate h_t ---
    h_t = compute_hardness_gate(ce_t, shift_mask, tau_p, T_p)

    # --- Final weight ---
    w_t = w_min + (1.0 - w_min) * (1.0 - h_t)

    # --- Weighted CE loss ---
    weighted_ce = w_t * ce_t * shift_mask.float()
    num_valid = shift_mask.sum().float().clamp(min=1.0)
    loss = weighted_ce.sum() / num_valid

    # Return intermediates for logging
    intermediates = {
        "ce_t": ce_t,           # (B, T-1)
        "shift_mask": shift_mask,  # (B, T-1)
        "h_t": h_t,            # (B, T-1)
        "w_t": w_t,            # (B, T-1)
    }

    return loss, intermediates
