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
Core loss functions for Reliability-Calibrated Conflict-Aware Trust-Region Fine-Tuning.

The method uses two token-level signals:
  1. Conflict score (α_t): whether the supervision challenges the prior.
  2. Reliability score (r_t): whether the prior is stable and trustworthy.

These combine into a trust-region objective:
  L_t = α_t · CE_t + λ · g(r_t) · KL(p_θ ∥ p_0)_t
"""

import torch
import torch.nn.functional as F


def compute_conflict_score(
    frozen_logits: torch.Tensor,
    labels: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute token-level conflict score α_t.

    The conflict score measures how much the ground-truth supervision
    deviates from the frozen model's prior. High conflict = the prior
    strongly disagrees with the label.

    Components:
      - Surprisal: c_t = -log p_0(y_t)
      - Margin: m_t = log p_0(ŷ_t^(1)) - log p_0(y_t)
      - Full: c_t^full = λ1 * surprisal + λ2 * margin
      - Normalize and sigmoid → α_t ∈ [0, 1]

    Args:
        frozen_logits: Logits from frozen model, shape (B, T, V).
        labels: Ground-truth token ids, shape (B, T). -100 = ignore.
        lambda1: Weight for the surprisal component.
        lambda2: Weight for the margin component.
        tau: Temperature for the sigmoid mapping.
        eps: Small constant for numerical stability.

    Returns:
        alpha_t: Conflict scores, shape (B, T), values in [0, 1].
    """
    # Compute log probabilities from frozen model
    frozen_log_probs = F.log_softmax(frozen_logits, dim=-1)  # (B, T, V)

    # Create mask for valid tokens
    valid_mask = labels != -100  # (B, T)
    safe_labels = labels.clamp(min=0)  # replace -100 with 0 for indexing

    # Surprisal: -log p_0(y_t)
    gt_log_probs = frozen_log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)
    surprisal = -gt_log_probs  # (B, T)

    # Margin: log p_0(ŷ^(1)) - log p_0(y_t)
    top1_log_probs = frozen_log_probs.max(dim=-1).values  # (B, T)
    margin = top1_log_probs - gt_log_probs  # (B, T)

    # Combined conflict score
    conflict_raw = lambda1 * surprisal + lambda2 * margin  # (B, T)

    # Normalize across valid tokens (z-score normalization)
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


def compute_stability(
    frozen_logits_perturbations: list[torch.Tensor],
    frozen_logits_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Compute stability-based reliability via Jensen-Shannon divergence
    between perturbed and reference frozen model outputs.

    s_t = 1 - (1/K) * Σ_k JSD(p_0^(k), p_0)

    High stability → the prior is robust to perturbations → likely real knowledge.
    Low stability → the prior is fragile → likely hallucination.

    Args:
        frozen_logits_perturbations: List of K logit tensors from perturbed
            frozen model, each shape (B, T, V).
        frozen_logits_ref: Reference frozen model logits, shape (B, T, V).

    Returns:
        s_t: Stability scores, shape (B, T), values in [0, 1].
    """
    ref_probs = F.softmax(frozen_logits_ref, dim=-1)  # (B, T, V)
    ref_log_probs = F.log_softmax(frozen_logits_ref, dim=-1)

    total_jsd = torch.zeros(
        ref_probs.shape[0], ref_probs.shape[1],
        device=ref_probs.device, dtype=ref_probs.dtype,
    )

    for pert_logits in frozen_logits_perturbations:
        pert_probs = F.softmax(pert_logits, dim=-1)
        pert_log_probs = F.log_softmax(pert_logits, dim=-1)

        # M = 0.5 * (p + q) for JSD
        m_probs = 0.5 * (ref_probs + pert_probs)
        m_log_probs = m_probs.clamp(min=1e-10).log()

        # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)
        kl_ref_m = (ref_probs * (ref_log_probs - m_log_probs)).sum(dim=-1)
        kl_pert_m = (pert_probs * (pert_log_probs - m_log_probs)).sum(dim=-1)
        jsd = 0.5 * kl_ref_m + 0.5 * kl_pert_m  # (B, T)

        total_jsd = total_jsd + jsd

    avg_jsd = total_jsd / max(len(frozen_logits_perturbations), 1)

    # s_t = 1 - avg_jsd, clamped to [0, 1]
    stability = (1.0 - avg_jsd).clamp(0.0, 1.0)

    return stability


def compute_evidence_drift(
    ema_logits: torch.Tensor,
    frozen_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute evidence-updated reliability.

    e_t = 1 - exp(-KL(p_ema || p_0)_t)
    r_t^evi = 1 - e_t = exp(-KL(p_ema || p_0)_t)

    When the EMA model has drifted far from the frozen prior,
    KL is large → r_t^evi is small → the prior is outdated.

    Args:
        ema_logits: Logits from EMA model, shape (B, T, V).
        frozen_logits: Logits from frozen model, shape (B, T, V).

    Returns:
        r_evi: Evidence-based reliability, shape (B, T), values in [0, 1].
    """
    ema_log_probs = F.log_softmax(ema_logits, dim=-1)
    frozen_log_probs = F.log_softmax(frozen_logits, dim=-1)
    ema_probs = F.softmax(ema_logits, dim=-1)

    # KL(p_ema || p_0) = Σ p_ema * (log p_ema - log p_0)
    kl_div = (ema_probs * (ema_log_probs - frozen_log_probs)).sum(dim=-1)  # (B, T)
    kl_div = kl_div.clamp(min=0.0)  # ensure non-negative

    # r_t^evi = exp(-KL)
    r_evi = torch.exp(-kl_div)

    return r_evi


def compute_reliability(
    stability: torch.Tensor,
    evidence_reliability: torch.Tensor,
    beta: float = 0.5,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Compute combined reliability score.

    r_t = σ((β * s_t + (1-β) * r_t^evi - μ_r) / τ_r)

    High reliability → prior is stable AND not contradicted by evidence.
    Low reliability → prior is either fragile or outdated by new data.

    Args:
        stability: Stability scores s_t, shape (B, T).
        evidence_reliability: Evidence reliability r_t^evi, shape (B, T).
        beta: Balance between stability and evidence. Default: 0.5.
        tau: Temperature for sigmoid. Default: 1.0.

    Returns:
        r_t: Combined reliability scores, shape (B, T), values in [0, 1].
    """
    combined = beta * stability + (1.0 - beta) * evidence_reliability  # (B, T)

    # Center and normalize
    mu = combined.mean()
    normalized = (combined - mu) / tau

    r_t = torch.sigmoid(normalized)

    return r_t


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
    Compute the trust-region fine-tuning objective.

    Smooth form:
        L_t = α_t · CE_t + λ · g(r_t) · KL(p_θ ∥ p_0)_t

    Hinge form:
        L_t = α_t · CE_t + λ · max(0, KL(p_θ ∥ p_0)_t - ε_t)
        where ε_t = ε_max - r_t * (ε_max - ε_min)

    In both cases:
      - α_t (conflict) controls whether to learn from this token.
      - r_t (reliability) controls how far the model can stray from the prior.

    Args:
        active_logits: Logits from active model, shape (B, T, V).
        frozen_logits: Logits from frozen model, shape (B, T, V).
        labels: Ground-truth token ids, shape (B, T). -100 = ignore.
        alpha_t: Conflict scores, shape (B, T).
        r_t: Reliability scores, shape (B, T).
        kl_lambda: Lagrange multiplier for KL term.
        epsilon_min: Minimum trust-region radius (hinge only).
        epsilon_max: Maximum trust-region radius (hinge only).
        use_smooth: If True, use smooth form; otherwise, hinge form.
        num_items_in_batch: For sample packing normalization.

    Returns:
        Scalar loss tensor.
    """
    valid_mask = labels != -100  # (B, T)
    safe_labels = labels.clamp(min=0)

    # --- Cross-entropy loss (per token) ---
    # Shift logits and labels for next-token prediction
    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_safe_labels = shift_labels.clamp(min=0)
    shift_mask = shift_labels != -100
    shift_alpha = alpha_t[..., 1:].contiguous()
    shift_r = r_t[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    # Per-token CE loss
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_ce = ce_loss_fn(
        shift_logits.view(-1, vocab_size),
        shift_safe_labels.view(-1),
    ).view(shift_labels.shape)  # (B, T-1)

    # Weighted by conflict score
    weighted_ce = shift_alpha * per_token_ce * shift_mask.float()  # (B, T-1)

    # --- KL divergence KL(p_θ ∥ p_0) per token ---
    shift_frozen_logits = frozen_logits[..., :-1, :].contiguous()

    active_log_probs = F.log_softmax(shift_logits, dim=-1)
    frozen_log_probs = F.log_softmax(shift_frozen_logits, dim=-1)
    active_probs = F.softmax(shift_logits, dim=-1)

    # KL(p_θ ∥ p_0) = Σ p_θ * (log p_θ - log p_0)
    per_token_kl = (active_probs * (active_log_probs - frozen_log_probs)).sum(
        dim=-1
    )  # (B, T-1)
    per_token_kl = per_token_kl.clamp(min=0.0)

    if use_smooth:
        # Smooth Lagrangian: λ · g(r_t) · KL_t
        # g(r_t) = r_t (monotonically increasing: more reliable → stronger KL penalty)
        kl_term = kl_lambda * shift_r * per_token_kl * shift_mask.float()
    else:
        # Hinge form: λ · max(0, KL_t - ε_t)
        epsilon_t = epsilon_max - shift_r * (epsilon_max - epsilon_min)
        kl_hinge = torch.clamp(per_token_kl - epsilon_t, min=0.0)
        kl_term = kl_lambda * kl_hinge * shift_mask.float()

    # --- Combine ---
    total_per_token = weighted_ce + kl_term  # (B, T-1)

    if num_items_in_batch is not None:
        loss = total_per_token.sum() / num_items_in_batch
    else:
        num_valid = shift_mask.float().sum().clamp(min=1.0)
        loss = total_per_token.sum() / num_valid

    return loss
