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
Core loss functions for RCCA-TR A+ variant.

A+ uses cached prior values instead of live frozen model forward passes:
  - Conflict score from cached prior_target_logp and prior_margin
  - Drift-based evidence reliability instead of EMA model comparison
  - Trust-region loss with KL proxy from cached prior logp

The method uses two token-level signals:
  1. Conflict score (α_t): whether the supervision challenges the prior.
  2. Reliability score (r_t): whether the prior is stable and trustworthy.

These combine into a trust-region objective:
  L_t = α_t · CE_t + λ · g(r_t) · CE_t  (weighted form)
"""

import torch
import torch.nn.functional as F


def compute_conflict_score_from_cache(
    prior_target_logp: torch.Tensor,
    prior_margin: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute token-level conflict score α_t from cached prior values.

    c_t^full = λ1 * (-log p_0(y_t)) + λ2 * margin_t
    α_t = σ((c_t - μ) / τ)

    Args:
        prior_target_logp: Cached log p_0(y_t), shape (B, T).
        prior_margin: Cached log p_0(ŷ^(1)) - log p_0(y_t), shape (B, T).
        valid_mask: Boolean mask for valid tokens, shape (B, T).
        lambda1: Weight for surprisal component.
        lambda2: Weight for margin component.
        tau: Temperature for sigmoid mapping.
        eps: Numerical stability constant.

    Returns:
        alpha_t: Conflict scores in [0, 1], shape (B, T).
    """
    # Surprisal = -log p_0(y_t)
    surprisal = -prior_target_logp  # (B, T)

    # Combined conflict
    conflict_raw = lambda1 * surprisal + lambda2 * prior_margin  # (B, T)

    # Normalize across valid tokens (z-score)
    if valid_mask.any():
        valid_values = conflict_raw[valid_mask]
        mu = valid_values.mean()
        sigma = valid_values.std() + eps
        conflict_normalized = (conflict_raw - mu) / sigma
    else:
        conflict_normalized = conflict_raw

    # Map to [0, 1] via sigmoid
    alpha_t = torch.sigmoid(conflict_normalized / tau)

    # Zero out invalid positions
    alpha_t = alpha_t * valid_mask.float()

    return alpha_t


def compute_reliability_from_drift(
    drift: torch.Tensor,
    gamma: float = 1.0,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Compute combined reliability from drift buffer.

    In A+, stability is not computed (no perturbation passes).
    Reliability comes purely from evidence drift:
        r_evi = exp(-γ * d_t)
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


def compute_trust_region_loss_cached(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha_t: torch.Tensor,
    r_t: torch.Tensor,
    prior_target_logp: torch.Tensor,
    kl_lambda: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_max: float = 1.0,
    use_smooth: bool = True,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute trust-region loss using cached prior values (A+ variant).

    Smooth form:
        L_t = α_t · CE_t + λ · g(r_t) · KL_proxy_t

    Where KL_proxy_t = CE(p_θ, y_t) - (-prior_target_logp)
                     = -log p_θ(y_t) + log p_0(y_t)

    This is the difference in surprisal between the active and frozen models,
    which approximates the KL divergence at the target token.

    Args:
        active_logits: Logits from active model, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T). -100 = ignore.
        alpha_t: Conflict scores, shape (B, T).
        r_t: Reliability scores, shape (B, T).
        prior_target_logp: Cached log p_0(y_t), shape (B, T).
        kl_lambda: Lagrange multiplier for KL term.
        epsilon_min: Min trust-region radius (hinge only).
        epsilon_max: Max trust-region radius (hinge only).
        use_smooth: If True, smooth form; otherwise hinge.
        num_items_in_batch: For sample packing normalization.

    Returns:
        Tuple of (scalar loss, active_target_logp for drift buffer update).
    """
    # Shift for next-token prediction
    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_safe_labels = shift_labels.clamp(min=0)
    shift_mask = shift_labels != -100
    shift_alpha = alpha_t[..., 1:].contiguous()
    shift_r = r_t[..., 1:].contiguous()
    shift_prior_logp = prior_target_logp[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    # Per-token CE loss
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_ce = ce_loss_fn(
        shift_logits.view(-1, vocab_size),
        shift_safe_labels.view(-1),
    ).view(shift_labels.shape)  # (B, T-1)

    # Weighted by conflict score
    weighted_ce = shift_alpha * per_token_ce * shift_mask.float()

    # KL proxy: -log p_θ(y_t) - (-log p_0(y_t)) = CE_θ - surprisal_0
    # = per_token_ce + prior_target_logp (since prior_target_logp = log p_0)
    kl_proxy = (per_token_ce + shift_prior_logp).clamp(min=0.0)  # (B, T-1)

    if use_smooth:
        # Smooth: λ · g(r_t) · KL_proxy
        kl_term = kl_lambda * shift_r * kl_proxy * shift_mask.float()
    else:
        # Hinge: λ · max(0, KL_proxy - ε_t)
        epsilon_t = epsilon_max - shift_r * (epsilon_max - epsilon_min)
        kl_hinge = torch.clamp(kl_proxy - epsilon_t, min=0.0)
        kl_term = kl_lambda * kl_hinge * shift_mask.float()

    # Combine
    total_per_token = weighted_ce + kl_term

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


# ===== Legacy functions kept for backward compatibility =====


def compute_conflict_score(
    frozen_logits: torch.Tensor,
    labels: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute conflict score from full frozen logits (legacy, for testing).
    """
    frozen_log_probs = F.log_softmax(frozen_logits, dim=-1)
    valid_mask = labels != -100
    safe_labels = labels.clamp(min=0)

    gt_log_probs = frozen_log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    surprisal = -gt_log_probs
    top1_log_probs = frozen_log_probs.max(dim=-1).values
    margin = top1_log_probs - gt_log_probs

    conflict_raw = lambda1 * surprisal + lambda2 * margin

    if valid_mask.any():
        valid_values = conflict_raw[valid_mask]
        mu = valid_values.mean()
        sigma = valid_values.std() + eps
        conflict_normalized = (conflict_raw - mu) / sigma
    else:
        conflict_normalized = conflict_raw

    alpha_t = torch.sigmoid(conflict_normalized / tau)
    alpha_t = alpha_t * valid_mask.float()
    return alpha_t


def compute_trust_region_loss(
    active_logits: torch.Tensor,
    frozen_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha_t: torch.Tensor,
    r_t: torch.Tensor,
    kl_lambda: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_max: float = 1.0,
    use_smooth: bool = True,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Legacy trust-region loss with full frozen logits (for backward compat / testing).
    """
    valid_mask = labels != -100
    safe_labels = labels.clamp(min=0)

    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_safe_labels = shift_labels.clamp(min=0)
    shift_mask = shift_labels != -100
    shift_alpha = alpha_t[..., 1:].contiguous()
    shift_r = r_t[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_ce = ce_loss_fn(
        shift_logits.view(-1, vocab_size),
        shift_safe_labels.view(-1),
    ).view(shift_labels.shape)

    weighted_ce = shift_alpha * per_token_ce * shift_mask.float()

    shift_frozen_logits = frozen_logits[..., :-1, :].contiguous()
    active_log_probs = F.log_softmax(shift_logits, dim=-1)
    frozen_log_probs = F.log_softmax(shift_frozen_logits, dim=-1)
    active_probs = F.softmax(shift_logits, dim=-1)

    per_token_kl = (active_probs * (active_log_probs - frozen_log_probs)).sum(dim=-1)
    per_token_kl = per_token_kl.clamp(min=0.0)

    if use_smooth:
        kl_term = kl_lambda * shift_r * per_token_kl * shift_mask.float()
    else:
        epsilon_t = epsilon_max - shift_r * (epsilon_max - epsilon_min)
        kl_hinge = torch.clamp(per_token_kl - epsilon_t, min=0.0)
        kl_term = kl_lambda * kl_hinge * shift_mask.float()

    total_per_token = weighted_ce + kl_term

    if num_items_in_batch is not None:
        loss = total_per_token.sum() / num_items_in_batch
    else:
        num_valid = shift_mask.float().sum().clamp(min=1.0)
        loss = total_per_token.sum() / num_valid

    return loss
