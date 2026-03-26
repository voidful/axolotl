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
Drift: Unified Risk Score Loss.

Combines instantaneous hardness (CE_t) and historical drift (d_t)
into a single token risk z_t, then maps to reliability r_t.

  CE_t = -log p_θ(y_t)                       (instantaneous hardness)
  d_t  = ρ · d_{t-1} + (1-ρ) · CE_t          (historical drift)
  z_t  = β · zn(CE_t) + (1-β) · zn(d_t)      (unified risk, zn = z-score)
  r_t  = σ(-z_t / τ)                          (reliability)
  L    = (λ / |V|) · Σ r_t · CE_t             (loss)

Intuition:
  - CE_t high + d_t high → dangerous token, suppress hard
  - CE_t high + d_t low  → temporarily hard, being learned, don't suppress
  - CE_t low  + d_t low  → easy, learn normally (r_t ≈ high)
"""

import torch
import torch.nn.functional as F


def compute_unified_risk(
    ce_t: torch.Tensor,
    d_t: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float = 0.5,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute unified risk score and map to reliability.

    z_t = β · (CE_t - μ_CE) / σ_CE + (1-β) · (d_t - μ_d) / σ_d
    r_t = σ(-z_t / τ)

    Args:
        ce_t: Per-token cross-entropy, shape (B, T).
        d_t: Per-token drift, shape (B, T).
        valid_mask: Boolean mask, shape (B, T).
        beta: Balance between hardness (CE) and drift.
        tau: Temperature for sigmoid.
        eps: Numerical stability.

    Returns:
        r_t: Reliability scores in [0, 1], shape (B, T).
    """
    mask_f = valid_mask.float()

    # Z-score normalize CE over valid tokens
    valid_ce = ce_t[valid_mask]
    mu_ce = valid_ce.mean() if valid_ce.numel() > 0 else ce_t.new_tensor(0.0)
    sigma_ce = valid_ce.std() if valid_ce.numel() > 1 else ce_t.new_tensor(1.0)
    sigma_ce = sigma_ce.clamp(min=eps)
    z_ce = (ce_t - mu_ce) / sigma_ce

    # Z-score normalize drift over valid tokens
    valid_d = d_t[valid_mask]
    mu_d = valid_d.mean() if valid_d.numel() > 0 else d_t.new_tensor(0.0)
    sigma_d = valid_d.std() if valid_d.numel() > 1 else d_t.new_tensor(1.0)
    sigma_d = sigma_d.clamp(min=eps)
    z_d = (d_t - mu_d) / sigma_d

    # Unified risk
    z_t = beta * z_ce + (1.0 - beta) * z_d

    # Map to reliability: high risk → low reliability
    r_t = torch.sigmoid(-z_t / tau)

    return r_t * mask_f


def compute_risk_weighted_loss(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    r_t: torch.Tensor,
    lam: float = 1.0,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute risk-weighted CE loss.

    L = (λ / |V|) · Σ r_t · CE_t

    Args:
        active_logits: Model logits, shape (B, T, V).
        labels: Token IDs, shape (B, T). -100 = ignore.
        r_t: Reliability scores, shape (B, T).
        lam: Overall loss multiplier.
        num_items_in_batch: For sample packing normalization.

    Returns:
        Tuple of (scalar loss, per-token CE for drift update).
    """
    # Shift for next-token prediction
    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_safe_labels = shift_labels.clamp(min=0)
    shift_mask = shift_labels != -100
    # r_t[t] = risk of label[t], so for shifted loss predicting label[t+1],
    # we need r_t[1:] to match shift_labels = labels[1:]
    shift_r = r_t[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    # Per-token CE
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_ce = ce_loss_fn(
        shift_logits.view(-1, vocab_size),
        shift_safe_labels.view(-1),
    ).view(shift_labels.shape)  # (B, T-1)

    # Weighted loss: λ · r_t · CE_t
    weighted = lam * shift_r * per_token_ce * shift_mask.float()

    if num_items_in_batch is not None:
        loss = weighted.sum() / num_items_in_batch
    else:
        num_valid = shift_mask.float().sum().clamp(min=1.0)
        loss = weighted.sum() / num_valid

    return loss, per_token_ce
