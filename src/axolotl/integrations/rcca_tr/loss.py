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
Unified loss dispatch for Drift-Trust and token-role triage.

Legacy modes:
  ce            — Standard cross-entropy (baseline).
  hardness      — Self-paced hardness weighting: w_t = w_min + (1-w_min)·s_t
  drift_trust_s — Suppressive temporal drift weighting.
  drift_trust_a — Anchoring temporal drift weighting.

Base-aware modes:
  stm_top20               — Mask the top 20% highest-base-NLL tokens.
  soft_stm                — Soft low-base-NLL token learning.
  retention_kl            — Downweight old-known tokens and add base KL.
  learn_new               — Upweight trustworthy new tokens, suppress outliers.
  module_aware_retention  — Full-FT module-routed token-role retention.
  fullft_module_aware_retention — Explicit alias for the full-FT routed method.
  attention_only_new      — Acquisition-focused token triage ablation.
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


def _target_logp_from_logits(
    shift_logits: torch.Tensor,
    shift_safe_labels: torch.Tensor,
) -> torch.Tensor:
    """Compute log p(y_t) without materializing the full log-prob tensor."""
    target_logit = shift_logits.gather(
        dim=-1,
        index=shift_safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    log_z = torch.logsumexp(shift_logits, dim=-1)
    return target_logit - log_z


def _masked_quantile(
    values: torch.Tensor,
    mask: torch.Tensor,
    q: float,
) -> torch.Tensor:
    """Return a detached quantile over valid positions."""
    q = min(max(float(q), 0.0), 1.0)
    if mask.any():
        return torch.quantile(values[mask].float().detach(), q).to(
            device=values.device,
            dtype=values.dtype,
        )
    return values.detach().float().mean().to(device=values.device, dtype=values.dtype)


def _resolve_threshold(
    values: torch.Tensor,
    mask: torch.Tensor,
    threshold: float | None,
    quantile: float,
) -> torch.Tensor:
    """Use an explicit threshold when provided, otherwise a batch quantile."""
    if threshold is not None:
        return values.new_tensor(float(threshold))
    return _masked_quantile(values, mask, quantile)


def _per_token_forward_kl(
    base_shift_logits: torch.Tensor,
    active_shift_logits: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Compute KL(p_base || p_active) per token.

    The computation is chunked along sequence length to avoid keeping two full
    log-softmax tensors for long-context 4B runs.
    """
    batch, seq_len = mask.shape
    kl_t = base_shift_logits.new_zeros((batch, seq_len), dtype=torch.float32)
    chunk_size = max(int(chunk_size), 1)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        base_chunk = base_shift_logits[:, start:end, :].float()
        active_chunk = active_shift_logits[:, start:end, :].float()
        base_logp = F.log_softmax(base_chunk, dim=-1)
        active_logp = F.log_softmax(active_chunk, dim=-1)
        kl_chunk = (base_logp.exp() * (base_logp - active_logp)).sum(dim=-1)
        kl_t[:, start:end] = kl_chunk

    return kl_t.to(device=base_shift_logits.device) * mask.float()


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


def compute_suppressive_drift_score(
    drift: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Suppressive drift score for Drift-Trust-S.

    r_t = σ((d_t − μ) / τ)

    Positive drift (easier than history) → r_t ≈ 1 (reliable, learn).
    Negative drift (harder than history) → r_t ≈ 0 (suspicious, suppress).

    Args:
        drift: Per-token signed drift from DriftBuffer, shape (B, T).
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


def compute_anchoring_drift_score(
    drift: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 1.0,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Anchoring drift score for Drift-Trust-A.

    r_evi = exp(-γ · |drift|)
    r_t   = σ((r_evi − μ) / τ)

    Low |drift| (stable token) → r_evi ≈ 1 → r_t high → amplify as anchor.
    High |drift| (changing token) → r_evi ≈ 0 → r_t low → normal learning.

    Args:
        drift: Per-token signed drift from DriftBuffer, shape (B, T).
        valid_mask: Boolean mask, shape (B, T).
        gamma: Decay rate for |drift| → evidence conversion.
        tau: Temperature for sigmoid normalization.

    Returns:
        r_t in [0, 1], shape (B, T).
    """
    r_evi = torch.exp(-gamma * drift.abs())

    if valid_mask.any():
        mu = r_evi[valid_mask].mean()
    else:
        mu = r_evi.mean()

    r_t = torch.sigmoid((r_evi - mu) / tau)
    return r_t * valid_mask.float()


def compute_token_role_gates(
    base_nll_t: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    tau_old: float | None = None,
    tau_new: float | None = None,
    tau_noise: float | None = None,
    gate_temperature: float = 1.0,
    old_quantile: float = 0.4,
    new_quantile: float = 0.6,
    noise_quantile: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute base-aware token role gates.

    R_t: old-known / preserve token gate, high when base NLL is low.
    A_t: acquisition gate, high for moderately high base NLL but not outliers.
    N_t: noise/outlier gate, high for extreme base NLL.
    """
    temperature = max(float(gate_temperature), 1e-6)
    tau_old_t = _resolve_threshold(base_nll_t, valid_mask, tau_old, old_quantile)
    tau_new_t = _resolve_threshold(base_nll_t, valid_mask, tau_new, new_quantile)
    tau_noise_t = _resolve_threshold(base_nll_t, valid_mask, tau_noise, noise_quantile)

    mask_f = valid_mask.float()
    r_t = torch.sigmoid((tau_old_t - base_nll_t) / temperature) * mask_f
    n_t = torch.sigmoid((base_nll_t - tau_noise_t) / temperature) * mask_f
    a_t = torch.sigmoid((base_nll_t - tau_new_t) / temperature) * (1.0 - n_t) * mask_f

    return r_t, a_t, n_t, {
        "tau_old": tau_old_t.detach(),
        "tau_new": tau_new_t.detach(),
        "tau_noise": tau_noise_t.detach(),
    }


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
    Self-paced hardness weighting (hardness-only ablation).

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


def compute_drift_trust_s_loss(
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
    Drift-Trust-S: Suppressive mapping.

    Best for noisy alignment — suppresses unreliable tokens.

    w_t = w_min + (1 - w_min) · (β · s_t + (1 - β) · r_t)
    L = Σ w_t · CE_t / N

    Weight range: [w_min, 1.0]

    Args:
        logits: Model output logits, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T). -100 = ignore.
        drift: Per-token signed drift from DriftBuffer, shape (B, T).
        self_tau: Temperature for self-paced score.
        drift_tau: Temperature for suppressive drift score.
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
    r_t = compute_suppressive_drift_score(shift_drift, mask, drift_tau)

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


def compute_drift_trust_a_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    drift: torch.Tensor,
    gamma: float = 1.0,
    reliability_tau: float = 1.0,
    anchor_base: float = 0.1,
    anchor_lambda: float = 4.0,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Drift-Trust-A: Anchoring mapping.

    Best for clean domain specialization — amplifies knowledge anchors.

    r_evi = exp(-γ · |drift|)
    r_t   = σ((r_evi - μ) / τ)
    w_t   = w_0 + λ · r_t
    L     = Σ w_t · CE_t / N

    Weight range: [w_0, w_0 + λ]  (default: [0.1, 4.1])

    Args:
        logits: Model output logits, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T). -100 = ignore.
        drift: Per-token signed drift from DriftBuffer, shape (B, T).
        gamma: Decay rate for |drift| → evidence conversion.
        reliability_tau: Temperature for anchoring score sigmoid.
        anchor_base: Base weight w_0 (minimum contribution).
        anchor_lambda: Amplification factor λ for reliable tokens.
        num_items_in_batch: For sample-packing normalization.

    Returns:
        Tuple of (scalar loss, stats dict for logging).
    """
    shift_logits, shift_labels, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)

    shift_drift = drift[..., 1:].contiguous()
    r_t = compute_anchoring_drift_score(shift_drift, mask, gamma, reliability_tau)

    # Anchoring: w_0 + λ · r_t
    w_t = anchor_base + anchor_lambda * r_t

    weighted_ce = w_t * ce_t * mask.float()
    loss = _reduce(weighted_ce, mask, num_items_in_batch)

    return loss, {
        "ce_t": ce_t.detach(),
        "r_t": r_t.detach(),
        "w_t": w_t.detach(),
    }


def _base_nll_from_logits(
    base_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return shifted base logits, mask, safe labels, and base NLL."""
    base_shift_logits, _, mask, safe_labels = _shift_and_mask(base_logits, labels)
    base_target_logp = _target_logp_from_logits(base_shift_logits, safe_labels)
    base_nll_t = (-base_target_logp) * mask.float()
    return base_shift_logits, mask, safe_labels, base_nll_t


def compute_stm_topk_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    base_logits: torch.Tensor,
    keep_ratio: float = 0.8,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Selective token masking: ignore the highest-base-NLL tail."""
    shift_logits, _, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)
    _, _, _, base_nll_t = _base_nll_from_logits(base_logits, labels)

    tau_keep = _masked_quantile(base_nll_t, mask, keep_ratio)
    w_t = (base_nll_t <= tau_keep).float() * mask.float()
    weighted_ce = w_t * ce_t * mask.float()
    loss = _reduce(weighted_ce, mask, num_items_in_batch)

    return loss, {
        "ce_t": ce_t.detach(),
        "base_nll_t": base_nll_t.detach(),
        "w_t": w_t.detach(),
        "tau_noise": tau_keep.detach(),
    }


def compute_soft_stm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    base_logits: torch.Tensor,
    tau_old: float | None = None,
    gate_temperature: float = 1.0,
    keep_ratio: float = 0.8,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft low-base-NLL token learning: w_t = sigmoid((tau - base_nll_t) / T)."""
    shift_logits, _, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)
    _, _, _, base_nll_t = _base_nll_from_logits(base_logits, labels)

    tau_old_t = _resolve_threshold(base_nll_t, mask, tau_old, keep_ratio)
    temperature = max(float(gate_temperature), 1e-6)
    w_t = torch.sigmoid((tau_old_t - base_nll_t) / temperature) * mask.float()
    weighted_ce = w_t * ce_t * mask.float()
    loss = _reduce(weighted_ce, mask, num_items_in_batch)

    return loss, {
        "ce_t": ce_t.detach(),
        "base_nll_t": base_nll_t.detach(),
        "w_t": w_t.detach(),
        "R_t": w_t.detach(),
        "tau_old": tau_old_t.detach(),
    }


def compute_token_triage_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    tau_old: float | None = None,
    tau_new: float | None = None,
    tau_noise: float | None = None,
    gate_temperature: float = 1.0,
    old_quantile: float = 0.4,
    new_quantile: float = 0.6,
    noise_quantile: float = 0.95,
    lambda_acquire: float = 1.0,
    mu_noise: float = 1.0,
    rho_retention: float = 0.5,
    w_floor: float = 0.1,
    w_max: float = 3.0,
    kl_beta: float = 0.05,
    kl_chunk_size: int = 256,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Base-aware token role triage loss.

    w_ce_t = clip(1 + λA_t - μN_t - ρR_t, w_floor, w_max)
    L = mean(w_ce_t * CE_t) + β mean(R_t * KL(base || active))
    """
    shift_logits, _, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)
    base_shift_logits, _, _, base_nll_t = _base_nll_from_logits(base_logits, labels)

    r_t, a_t, n_t, thresholds = compute_token_role_gates(
        base_nll_t,
        mask,
        tau_old=tau_old,
        tau_new=tau_new,
        tau_noise=tau_noise,
        gate_temperature=gate_temperature,
        old_quantile=old_quantile,
        new_quantile=new_quantile,
        noise_quantile=noise_quantile,
    )

    w_t = (
        1.0
        + float(lambda_acquire) * a_t
        - float(mu_noise) * n_t
        - float(rho_retention) * r_t
    )
    w_t = torch.clamp(w_t, min=float(w_floor), max=float(w_max)) * mask.float()
    ce_loss = _reduce(w_t * ce_t * mask.float(), mask, num_items_in_batch)

    if kl_beta > 0.0:
        kl_t = _per_token_forward_kl(
            base_shift_logits,
            shift_logits,
            mask,
            chunk_size=kl_chunk_size,
        )
        kl_loss = _reduce(r_t * kl_t * mask.float(), mask, num_items_in_batch)
        loss = ce_loss + float(kl_beta) * kl_loss
    else:
        kl_t = torch.zeros_like(ce_t)
        kl_loss = ce_t.new_zeros(())
        loss = ce_loss

    stats = {
        "ce_t": ce_t.detach(),
        "base_nll_t": base_nll_t.detach(),
        "R_t": r_t.detach(),
        "A_t": a_t.detach(),
        "N_t": n_t.detach(),
        "kl_t": kl_t.detach(),
        "w_t": w_t.detach(),
        "ce_loss": ce_loss.detach(),
        "kl_loss": kl_loss.detach(),
    }
    stats.update(thresholds)
    return loss, stats


def _role_weight(
    r_t: torch.Tensor,
    a_t: torch.Tensor,
    n_t: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_acquire: float,
    mu_noise: float,
    rho_retention: float,
    w_floor: float,
    w_max: float,
) -> torch.Tensor:
    """Build a clipped token-role CE weight for one parameter route."""
    w_t = (
        1.0
        + float(lambda_acquire) * a_t
        - float(mu_noise) * n_t
        - float(rho_retention) * r_t
    )
    return torch.clamp(w_t, min=float(w_floor), max=float(w_max)) * mask.float()


def _route_loss(
    ce_t: torch.Tensor,
    kl_t: torch.Tensor,
    r_t: torch.Tensor,
    mask: torch.Tensor,
    w_t: torch.Tensor,
    *,
    kl_beta: float,
    num_items_in_batch: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, CE, and KL loss for one parameter route."""
    ce_loss = _reduce(w_t * ce_t * mask.float(), mask, num_items_in_batch)
    if kl_beta > 0.0:
        kl_loss = _reduce(r_t * kl_t * mask.float(), mask, num_items_in_batch)
        total = ce_loss + float(kl_beta) * kl_loss
    else:
        kl_loss = ce_t.new_zeros(())
        total = ce_loss
    return total, ce_loss, kl_loss


def compute_module_routing_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    tau_old: float | None = None,
    tau_new: float | None = None,
    tau_noise: float | None = None,
    gate_temperature: float = 1.0,
    old_quantile: float = 0.4,
    new_quantile: float = 0.6,
    noise_quantile: float = 0.95,
    attn_lambda_acquire: float = 1.0,
    attn_mu_noise: float = 1.0,
    attn_rho_retention: float = 0.0,
    attn_kl_beta: float = 0.0,
    mlp_lambda_acquire: float = 0.5,
    mlp_mu_noise: float = 1.0,
    mlp_rho_retention: float = 0.5,
    mlp_kl_beta: float = 0.05,
    other_lambda_acquire: float = 0.25,
    other_mu_noise: float = 1.0,
    other_rho_retention: float = 0.75,
    other_kl_beta: float = 0.05,
    w_floor: float = 0.1,
    w_max: float = 3.0,
    kl_chunk_size: int = 256,
    num_items_in_batch: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Full-FT module-routed token-role retention objective.

    A single forward pass produces shared token roles from the frozen base:
      R_t: old-known tokens to preserve.
      A_t: new but non-outlier tokens to acquire.
      N_t: high-NLL outliers/noise to suppress.

    The trainer backpropagates each route only into its matching parameter
    group. Attention receives acquisition-heavy CE. MLP receives selective CE
    plus stronger retention KL. Other trainable parameters are conservative.
    """
    shift_logits, _, mask, safe_labels = _shift_and_mask(logits, labels)
    ce_t = _per_token_ce(shift_logits, safe_labels)
    base_shift_logits, _, _, base_nll_t = _base_nll_from_logits(base_logits, labels)

    r_t, a_t, n_t, thresholds = compute_token_role_gates(
        base_nll_t,
        mask,
        tau_old=tau_old,
        tau_new=tau_new,
        tau_noise=tau_noise,
        gate_temperature=gate_temperature,
        old_quantile=old_quantile,
        new_quantile=new_quantile,
        noise_quantile=noise_quantile,
    )

    needs_kl = any(
        beta > 0.0
        for beta in (attn_kl_beta, mlp_kl_beta, other_kl_beta)
    )
    if needs_kl:
        kl_t = _per_token_forward_kl(
            base_shift_logits,
            shift_logits,
            mask,
            chunk_size=kl_chunk_size,
        )
    else:
        kl_t = torch.zeros_like(ce_t)

    w_attn_t = _role_weight(
        r_t,
        a_t,
        n_t,
        mask,
        lambda_acquire=attn_lambda_acquire,
        mu_noise=attn_mu_noise,
        rho_retention=attn_rho_retention,
        w_floor=w_floor,
        w_max=w_max,
    )
    w_mlp_t = _role_weight(
        r_t,
        a_t,
        n_t,
        mask,
        lambda_acquire=mlp_lambda_acquire,
        mu_noise=mlp_mu_noise,
        rho_retention=mlp_rho_retention,
        w_floor=w_floor,
        w_max=w_max,
    )
    w_other_t = _role_weight(
        r_t,
        a_t,
        n_t,
        mask,
        lambda_acquire=other_lambda_acquire,
        mu_noise=other_mu_noise,
        rho_retention=other_rho_retention,
        w_floor=w_floor,
        w_max=w_max,
    )

    attn_loss, attn_ce_loss, attn_kl_loss = _route_loss(
        ce_t,
        kl_t,
        r_t,
        mask,
        w_attn_t,
        kl_beta=attn_kl_beta,
        num_items_in_batch=num_items_in_batch,
    )
    mlp_loss, mlp_ce_loss, mlp_kl_loss = _route_loss(
        ce_t,
        kl_t,
        r_t,
        mask,
        w_mlp_t,
        kl_beta=mlp_kl_beta,
        num_items_in_batch=num_items_in_batch,
    )
    other_loss, other_ce_loss, other_kl_loss = _route_loss(
        ce_t,
        kl_t,
        r_t,
        mask,
        w_other_t,
        kl_beta=other_kl_beta,
        num_items_in_batch=num_items_in_batch,
    )

    losses = {
        "loss": (attn_loss + mlp_loss + other_loss) / 3.0,
        "attn_loss": attn_loss,
        "mlp_loss": mlp_loss,
        "other_loss": other_loss,
    }

    stats = {
        "ce_t": ce_t.detach(),
        "base_nll_t": base_nll_t.detach(),
        "R_t": r_t.detach(),
        "A_t": a_t.detach(),
        "N_t": n_t.detach(),
        "kl_t": kl_t.detach(),
        "w_t": w_mlp_t.detach(),
        "w_attn_t": w_attn_t.detach(),
        "w_mlp_t": w_mlp_t.detach(),
        "w_other_t": w_other_t.detach(),
        "attn_loss": attn_loss.detach(),
        "attn_ce_loss": attn_ce_loss.detach(),
        "attn_kl_loss": attn_kl_loss.detach(),
        "mlp_loss": mlp_loss.detach(),
        "mlp_ce_loss": mlp_ce_loss.detach(),
        "mlp_kl_loss": mlp_kl_loss.detach(),
        "other_loss": other_loss.detach(),
        "other_ce_loss": other_ce_loss.detach(),
        "other_kl_loss": other_kl_loss.detach(),
    }
    stats.update(thresholds)
    return losses, stats


# ── Legacy drift aliases ────────────────────────────────────────────────
compute_drift_only_loss = compute_drift_trust_s_loss
compute_drift_legacy_loss = compute_drift_trust_a_loss


# ── Unified dispatch ───────────────────────────────────────────────────


def compute_rcca_loss(
    mode: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    drift: torch.Tensor | None = None,
    base_logits: torch.Tensor | None = None,
    *,
    self_tau: float = 1.0,
    drift_tau: float = 1.0,
    w_min: float = 0.05,
    beta: float = 0.5,
    gamma: float = 1.0,
    reliability_tau: float = 1.0,
    anchor_base: float = 0.1,
    anchor_lambda: float = 4.0,
    tau_old: float | None = None,
    tau_new: float | None = None,
    tau_noise: float | None = None,
    gate_temperature: float = 1.0,
    old_quantile: float = 0.4,
    new_quantile: float = 0.6,
    noise_quantile: float = 0.95,
    stm_keep_ratio: float = 0.8,
    lambda_acquire: float = 1.0,
    mu_noise: float = 1.0,
    rho_retention: float = 0.5,
    triage_w_floor: float = 0.1,
    triage_w_max: float = 3.0,
    kl_beta: float = 0.05,
    kl_chunk_size: int = 256,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Unified loss dispatch for all Drift-Trust modes.

    Args:
        mode: One of the legacy Drift-Trust or base-aware token-triage modes.
              Legacy aliases "drift_only" and "drift" are also accepted.
        logits: Model output logits, shape (B, T, V).
        labels: Ground-truth labels, shape (B, T).
        drift: Per-token signed drift from DriftBuffer (required for drift modes).
        base_logits: Frozen base logits (required for base-aware modes).
        **kwargs: Mode-specific hyperparameters.

    Returns:
        Tuple of (scalar loss, stats dict).

    Raises:
        ValueError: If mode is unknown or drift is missing.
    """
    # Resolve legacy aliases
    if mode == "drift_only":
        mode = "drift_trust_s"
    elif mode == "drift":
        mode = "drift_trust_a"

    if mode == "ce":
        return compute_ce_loss(logits, labels, num_items_in_batch)

    elif mode == "hardness":
        return compute_hardness_loss(
            logits, labels, self_tau, w_min, num_items_in_batch,
        )

    elif mode == "drift_trust_s":
        if drift is None:
            raise ValueError("drift_trust_s mode requires drift tensor from DriftBuffer")
        return compute_drift_trust_s_loss(
            logits, labels, drift,
            self_tau, drift_tau, w_min, beta, num_items_in_batch,
        )

    elif mode == "drift_trust_a":
        if drift is None:
            raise ValueError("drift_trust_a mode requires drift tensor from DriftBuffer")
        return compute_drift_trust_a_loss(
            logits, labels, drift,
            gamma, reliability_tau, anchor_base, anchor_lambda, num_items_in_batch,
        )

    elif mode == "stm_top20":
        if base_logits is None:
            raise ValueError("stm_top20 mode requires base_logits from frozen base")
        return compute_stm_topk_loss(
            logits,
            labels,
            base_logits,
            keep_ratio=stm_keep_ratio,
            num_items_in_batch=num_items_in_batch,
        )

    elif mode == "soft_stm":
        if base_logits is None:
            raise ValueError("soft_stm mode requires base_logits from frozen base")
        return compute_soft_stm_loss(
            logits,
            labels,
            base_logits,
            tau_old=tau_old,
            gate_temperature=gate_temperature,
            keep_ratio=stm_keep_ratio,
            num_items_in_batch=num_items_in_batch,
        )

    elif mode == "retention_kl":
        if base_logits is None:
            raise ValueError("retention_kl mode requires base_logits from frozen base")
        return compute_token_triage_loss(
            logits,
            labels,
            base_logits,
            tau_old=tau_old,
            tau_new=tau_new,
            tau_noise=tau_noise,
            gate_temperature=gate_temperature,
            old_quantile=old_quantile,
            new_quantile=new_quantile,
            noise_quantile=noise_quantile,
            lambda_acquire=0.0,
            mu_noise=0.0,
            rho_retention=rho_retention,
            w_floor=triage_w_floor,
            w_max=triage_w_max,
            kl_beta=kl_beta,
            kl_chunk_size=kl_chunk_size,
            num_items_in_batch=num_items_in_batch,
        )

    elif mode in {"learn_new", "attention_only_new"}:
        if base_logits is None:
            raise ValueError(f"{mode} mode requires base_logits from frozen base")
        return compute_token_triage_loss(
            logits,
            labels,
            base_logits,
            tau_old=tau_old,
            tau_new=tau_new,
            tau_noise=tau_noise,
            gate_temperature=gate_temperature,
            old_quantile=old_quantile,
            new_quantile=new_quantile,
            noise_quantile=noise_quantile,
            lambda_acquire=lambda_acquire,
            mu_noise=mu_noise,
            rho_retention=0.0,
            w_floor=triage_w_floor,
            w_max=triage_w_max,
            kl_beta=0.0,
            kl_chunk_size=kl_chunk_size,
            num_items_in_batch=num_items_in_batch,
        )

    elif mode in {"module_aware_retention", "fullft_module_aware_retention"}:
        if base_logits is None:
            raise ValueError("module_aware_retention mode requires base_logits from frozen base")
        losses, stats = compute_module_routing_losses(
            logits,
            labels,
            base_logits,
            tau_old=tau_old,
            tau_new=tau_new,
            tau_noise=tau_noise,
            gate_temperature=gate_temperature,
            old_quantile=old_quantile,
            new_quantile=new_quantile,
            noise_quantile=noise_quantile,
            attn_lambda_acquire=lambda_acquire,
            attn_mu_noise=mu_noise,
            attn_rho_retention=0.0,
            attn_kl_beta=0.0,
            mlp_lambda_acquire=max(lambda_acquire * 0.5, 0.0),
            mlp_mu_noise=mu_noise,
            mlp_rho_retention=rho_retention,
            mlp_kl_beta=kl_beta,
            other_lambda_acquire=max(lambda_acquire * 0.25, 0.0),
            other_mu_noise=mu_noise,
            other_rho_retention=max(rho_retention, 0.0),
            other_kl_beta=kl_beta,
            w_floor=triage_w_floor,
            w_max=triage_w_max,
            kl_chunk_size=kl_chunk_size,
            num_items_in_batch=num_items_in_batch,
        )
        return losses["loss"], stats

    else:
        raise ValueError(
            f"Unknown rcca_tr_mode: {mode!r}. "
            "Expected one of: ce, hardness, drift_trust_s, drift_trust_a, "
            "stm_top20, soft_stm, retention_kl, learn_new, "
            "module_aware_retention, fullft_module_aware_retention, attention_only_new"
        )
