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
Unified loss dispatch for RCCA-TR.

Four modes, one entry point:

  ce         — Standard cross-entropy (baseline).
  hardness   — Self-paced hardness weighting: w_t = w_min + (1-w_min)·s_t
  drift_only — Self-paced + drift:  w_t = w_min + (1-w_min)·(β·s_t + (1-β)·r_t)
  drift      — Legacy trust-region: L = (anchor + λ·r_t)·CE_t

Main method: ``drift_only``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── Shared primitives ──────────────────────────────────────────────────


def _shift_and_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift for next-token prediction, return (shift_logits, shift_labels, mask, safe_labels)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = shift_labels != -100
    shift_safe_labels = shift_labels.clamp(min=0)
    return shift_logits, shift_labels, shift_mask, shift_safe_labels


def _per_token_ce(
    shift_logits: torch.Tensor,
    shift_safe_labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token CE loss (no reduction)."""
    vocab_size = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_safe_labels.view(-1),
        reduction="none",
    ).view(shift_safe_labels.shape)


def _reduce(
    weighted_ce: torch.Tensor,
    mask: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """Reduce per-token loss to scalar."""
    if num_items_in_batch is not None:
        return weighted_ce.sum() / num_items_in_batch
    return weighted_ce.sum() / mask.float().sum().clamp(min=1.0)


# ── Score functions ────────────────────────────────────────────────────


def compute_self_paced_score(
    active_target_logp: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Self-paced (hardness) score.

    s_t = σ((log p_θ(y_t) − μ) / τ)

    High log-prob tokens → s_t ≈ 1 (model confident, learn freely).
    Low log-prob tokens  → s_t ≈ 0 (model uncertain, suppress).

    Args:
        active_target_logp: log p_θ(y_t), shape (B, T).
        valid_mask: Boolean mask, shape (B, T).
        tau: Temperature controlling sigmoid steepness.

    Returns:
        s_t in [0, 1], shape (B, T).
    """
    if valid_mask.any():
        mu = active_target_logp[valid_mask].mean()
    else:
        mu = active_target_logp.mean()

    s_t = torch.sigmoid((active_target_logp - mu) / tau)
    return s_t * valid_mask.float()


def compute_drift_score(
    drift: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Drift-based reliability score.

    r_t = σ((d_t − μ) / τ)

    Positive drift (easier than history) → r_t ≈ 1 (reliable, learn).
    Negative drift (harder than history) → r_t ≈ 0 (suspicious, suppress).

    Args:
        drift: Per-token drift from DriftBuffer, shape (B, T).
        valid_mask: Boolean mask, shape (B, T).
        tau: Temperature for sigmoid normalization.

    Returns:
        r_t in [0, 1], shape (B, T).
    """
    if valid_mask.any():
        mu = drift[valid_mask].mean()
    else:
        mu = drift.mean()

    r_t = torch.sigmoid((drift - mu) / tau)
    return r_t * valid_mask.float()


# ── Mode-specific loss functions ───────────────────────────────────────


def compute_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Standard cross-entropy (baseline mode)."""
    shift_logits, shift_labels, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)
    loss = _reduce(ce_t * mask.float(), mask, num_items_in_batch)

    return loss, {
        "ce_t": ce_t.detach(),
        "w_t": mask.float(),  # uniform weight = 1
    }


def compute_hardness_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    self_tau: float = 1.0,
    w_min: float = 0.05,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Self-paced hardness weighting (hardness-only mode).

    w_t = w_min + (1 - w_min) · s_t
    L = Σ w_t · CE_t / N
    """
    shift_logits, shift_labels, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)

    # Compute log-prob for self-paced score
    log_probs = F.log_softmax(shift_logits, dim=-1)
    active_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    active_logp = active_logp * mask.float()

    s_t = compute_self_paced_score(active_logp, mask, self_tau)
    w_t = w_min + (1.0 - w_min) * s_t

    weighted_ce = w_t * ce_t * mask.float()
    loss = _reduce(weighted_ce, mask, num_items_in_batch)

    return loss, {
        "ce_t": ce_t.detach(),
        "s_t": s_t.detach(),
        "w_t": w_t.detach(),
        "active_target_logp": active_logp.detach(),
    }


def compute_drift_only_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    drift: torch.Tensor,
    self_tau: float = 1.0,
    drift_tau: float = 1.0,
    w_min: float = 0.05,
    beta: float = 0.5,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Self-paced + drift regularization (main method).

    w_t = w_min + (1 - w_min) · (β · s_t + (1 - β) · r_t)
    L = Σ w_t · CE_t / N

    Args:
        logits: Model output logits, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T). -100 = ignore.
        drift: Per-token drift from DriftBuffer, shape (B, T).
        self_tau: Temperature for self-paced score.
        drift_tau: Temperature for drift score.
        w_min: Minimum weight floor (prevents full suppression).
        beta: Balance between self-paced (β) and drift (1-β).
        num_items_in_batch: For sample-packing normalization.

    Returns:
        Tuple of (scalar loss, stats dict for logging).
    """
    shift_logits, shift_labels, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)

    # Log-prob for self-paced score
    log_probs = F.log_softmax(shift_logits, dim=-1)
    active_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    active_logp = active_logp * mask.float()

    # Shift drift to match shifted labels
    shift_drift = drift[..., 1:].contiguous()

    s_t = compute_self_paced_score(active_logp, mask, self_tau)
    r_t = compute_drift_score(shift_drift, mask, drift_tau)

    w_t = w_min + (1.0 - w_min) * (beta * s_t + (1.0 - beta) * r_t)

    weighted_ce = w_t * ce_t * mask.float()
    loss = _reduce(weighted_ce, mask, num_items_in_batch)

    return loss, {
        "ce_t": ce_t.detach(),
        "s_t": s_t.detach(),
        "r_t": r_t.detach(),
        "w_t": w_t.detach(),
        "active_target_logp": active_logp.detach(),
    }


def compute_drift_legacy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    drift: torch.Tensor,
    gamma: float = 1.0,
    reliability_tau: float = 1.0,
    kl_lambda: float = 4.0,
    anchor_weight: float = 0.1,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Legacy drift trust-region loss (ablation mode).

    r_evi = exp(-γ · |drift|)
    r_t   = σ((r_evi - μ) / τ)
    L     = (anchor + λ · r_t) · CE_t

    This is the original formulation from the Drift integration,
    kept for ablation comparison.
    """
    shift_logits, shift_labels, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)

    shift_drift = drift[..., 1:].contiguous()
    r_evi = torch.exp(-gamma * shift_drift.abs())

    mu = r_evi.mean()
    r_t = torch.sigmoid((r_evi - mu) / reliability_tau)

    # Trust-region: (anchor + λ · r_t) · CE
    effective_weight = anchor_weight + kl_lambda * r_t
    weighted_ce = effective_weight * ce_t * mask.float()
    loss = _reduce(weighted_ce, mask, num_items_in_batch)

    return loss, {
        "ce_t": ce_t.detach(),
        "r_t": r_t.detach(),
        "w_t": effective_weight.detach(),
    }


# ── Unified dispatch ───────────────────────────────────────────────────


def compute_rcca_loss(
    mode: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    drift: torch.Tensor | None = None,
    *,
    self_tau: float = 1.0,
    drift_tau: float = 1.0,
    w_min: float = 0.05,
    beta: float = 0.5,
    gamma: float = 1.0,
    reliability_tau: float = 1.0,
    kl_lambda: float = 4.0,
    anchor_weight: float = 0.1,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Unified loss dispatch for all RCCA-TR modes.

    Args:
        mode: One of {"ce", "hardness", "drift_only", "drift"}.
        logits: Model output logits, shape (B, T, V).
        labels: Ground-truth labels, shape (B, T).
        drift: Per-token drift from DriftBuffer (required for drift modes).
        **kwargs: Mode-specific hyperparameters.

    Returns:
        Tuple of (scalar loss, stats dict).

    Raises:
        ValueError: If mode is unknown.
    """
    if mode == "ce":
        return compute_ce_loss(logits, labels, num_items_in_batch)

    elif mode == "hardness":
        return compute_hardness_loss(
            logits, labels, self_tau, w_min, num_items_in_batch,
        )

    elif mode == "drift_only":
        if drift is None:
            raise ValueError("drift_only mode requires drift tensor from DriftBuffer")
        return compute_drift_only_loss(
            logits, labels, drift,
            self_tau, drift_tau, w_min, beta, num_items_in_batch,
        )

    elif mode == "drift":
        if drift is None:
            raise ValueError("drift mode requires drift tensor from DriftBuffer")
        return compute_drift_legacy_loss(
            logits, labels, drift,
            gamma, reliability_tau, kl_lambda, anchor_weight, num_items_in_batch,
        )

    else:
        raise ValueError(
            f"Unknown rcca_tr_mode: {mode!r}. "
            f"Expected one of: ce, hardness, drift_only, drift"
        )
