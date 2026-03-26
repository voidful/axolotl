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
Core loss functions for RCCA-TR (no-cache variant).

Uses the active model's CE as the drift signal.
No frozen prior cache is needed.

Token-level signal:
  - Reliability score (r_t): drift-based evidence reliability.
    r_evi = exp(-γ · d_t), then z-score + sigmoid.

Trust-region objective:
  L_t = λ · r_t · CE_t   (smooth form)
"""

import torch
import torch.nn.functional as F


def compute_reliability_from_drift(
    drift: torch.Tensor,
    gamma: float = 1.0,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Compute combined reliability from drift buffer.

    r_evi = exp(-γ · d_t)
    r_t = σ((r_evi - μ) / τ)

    Args:
        drift: Per-token drift values, shape (B, T).
        gamma: Scaling factor for drift → reliability.
        tau: Temperature for sigmoid normalization.

    Returns:
        r_t: Reliability scores in [0, 1], shape (B, T).
    """
    r_evi = torch.exp(-gamma * drift)

    # Center and normalize
    mu = r_evi.mean()
    normalized = (r_evi - mu) / tau

    r_t = torch.sigmoid(normalized)

    return r_t


def compute_trust_region_loss(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    r_t: torch.Tensor,
    kl_lambda: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_max: float = 1.0,
    use_smooth: bool = True,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute trust-region loss (no-cache variant).

    Smooth form:
        L_t = λ · r_t · KL_proxy_t

    Without cache, KL_proxy = CE_t + 0 = CE_t, so:
        L_t = λ · r_t · CE_t

    Args:
        active_logits: Logits from active model, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T). -100 = ignore.
        r_t: Reliability scores, shape (B, T).
        kl_lambda: Lagrange multiplier (scales the overall loss).
        epsilon_min: Min trust-region radius (hinge only).
        epsilon_max: Max trust-region radius (hinge only).
        use_smooth: If True, smooth form; otherwise hinge.
        num_items_in_batch: For sample packing normalization.

    Returns:
        Tuple of (scalar loss, active_target_logp_shifted for drift buffer update).
    """
    # Shift for next-token prediction
    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_safe_labels = shift_labels.clamp(min=0)
    shift_mask = shift_labels != -100
    shift_r = r_t[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    # Per-token CE loss
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_ce = ce_loss_fn(
        shift_logits.view(-1, vocab_size),
        shift_safe_labels.view(-1),
    ).view(shift_labels.shape)  # (B, T-1)

    # KL proxy: CE_t (since prior_target_logp = 0)
    kl_proxy = per_token_ce.clamp(min=0.0)  # (B, T-1)

    if use_smooth:
        # Smooth: λ · r_t · KL_proxy
        total_per_token = kl_lambda * shift_r * kl_proxy * shift_mask.float()
    else:
        # Hinge: λ · max(0, KL_proxy - ε_t)
        epsilon_t = epsilon_max - shift_r * (epsilon_max - epsilon_min)
        kl_hinge = torch.clamp(kl_proxy - epsilon_t, min=0.0)
        total_per_token = kl_lambda * kl_hinge * shift_mask.float()

    if num_items_in_batch is not None:
        loss = total_per_token.sum() / num_items_in_batch
    else:
        num_valid = shift_mask.float().sum().clamp(min=1.0)
        loss = total_per_token.sum() / num_valid

    # Compute active model's log p(y_t) for drift buffer
    active_log_probs = F.log_softmax(shift_logits.detach(), dim=-1)
    active_target_logp_shifted = active_log_probs.gather(
        dim=-1, index=shift_safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    active_target_logp_shifted = active_target_logp_shifted * shift_mask.float()

    return loss, active_target_logp_shifted
