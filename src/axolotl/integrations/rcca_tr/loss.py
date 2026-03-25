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
Core loss functions for RCCA-TR: Suppress-by-Default, Rescue-if-Useful.

We suppress high-perplexity tokens by default, but rescue those that
consistently improve the model's confidence on the supervised target
relative to the frozen prior.

Two orthogonal token-level gates:
  1. Hardness gate (h_t): is this token hard enough to warrant suppression?
  2. Useful-hard gate (q_t): does this hard token have positive improvement evidence?

These combine into a single unified weight:
  w_t = w_min + (1 - w_min) · (1 - h_t · (1 - q_t))

Final loss:
  L = Σ w_t · CE_t / |V|
"""

import torch
import torch.nn.functional as F


# =====================================================================
# Gate computations
# =====================================================================


def compute_hardness_gate(
    ce_t: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_p: float = 2.0,
    T_p: float = 1.0,
) -> torch.Tensor:
    """
    Hardness gate: is this token hard enough to warrant default suppression?

    h_t = σ((u_t − τ_p) / T_p)    where u_t = CE_t

    High h_t → token is hard, should be suppressed by default.
    Low h_t  → token is easy, no suppression needed.

    Args:
        ce_t: Per-token cross-entropy, shape (B, T).
        valid_mask: Boolean mask for valid tokens, shape (B, T).
        tau_p: Hardness threshold (perplexity level above which suppression kicks in).
        T_p: Temperature for sigmoid smoothing.

    Returns:
        h_t: Hardness scores in [0, 1], shape (B, T).
    """
    h_t = torch.sigmoid((ce_t - tau_p) / T_p)
    return h_t * valid_mask.float()


def compute_useful_hard_gate(
    delta_plus: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_delta: float = 0.8,
    T_delta: float = 1.0,
) -> torch.Tensor:
    """
    Useful-hard gate: does this hard token provide positive improvement evidence?

    q_t = σ((Δ_t⁺ − τ_Δ) / T_Δ)

    where Δ_t⁺ = max(0, log p_θ(y_t) − log p₀(y_t))

    High q_t → active model is more confident than prior on gold token → rescue.
    Low q_t  → no improvement evidence → keep suppressed.

    Args:
        delta_plus: Directional improvement Δ_t⁺, shape (B, T).
        valid_mask: Boolean mask for valid tokens, shape (B, T).
        tau_delta: Improvement threshold.
        T_delta: Temperature for sigmoid smoothing.

    Returns:
        q_t: Useful-hard scores in [0, 1], shape (B, T).
    """
    q_t = torch.sigmoid((delta_plus - tau_delta) / T_delta)
    return q_t * valid_mask.float()


def compute_token_weights(
    h_t: torch.Tensor,
    q_t: torch.Tensor,
    w_min: float = 0.05,
) -> torch.Tensor:
    """
    Compute final per-token weight for CE loss.

    w_t = w_min + (1 − w_min) · (1 − h_t · (1 − q_t))

    Behavior:
      - Easy token (h≈0):           w ≈ 1
      - Hard + useless (h≈1, q≈0):  w ≈ w_min  (suppressed)
      - Hard + useful (h≈1, q≈1):   w ≈ 1      (rescued)

    Args:
        h_t: Hardness gate, shape (B, T).
        q_t: Useful-hard gate, shape (B, T).
        w_min: Weight floor to prevent zero gradients.

    Returns:
        w_t: Final per-token weights, shape (B, T).
    """
    return w_min + (1.0 - w_min) * (1.0 - h_t * (1.0 - q_t))


# =====================================================================
# Main loss function
# =====================================================================


def compute_weighted_ce_loss(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    prior_target_logp: torch.Tensor,
    tau_p: float = 2.0,
    T_p: float = 1.0,
    tau_delta: float = 0.8,
    T_delta: float = 1.0,
    w_min: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the RCCA-TR weighted CE loss: suppress-by-default, rescue-if-useful.

    L = Σ w_t · CE_t / |V|

    w_t = w_min + (1 − w_min) · (1 − h_t · (1 − q_t))

    All shifting for next-token prediction is handled internally.

    Args:
        active_logits: Logits from active model, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T). -100 = ignore.
        prior_target_logp: Cached log p₀(y_t), shape (B, T).
        tau_p: Hardness threshold.
        T_p: Hardness temperature.
        tau_delta: Improvement evidence threshold.
        T_delta: Improvement evidence temperature.
        w_min: Weight floor.

    Returns:
        Tuple of (scalar loss, dict of intermediate tensors for logging).
    """
    # Shift for next-token prediction
    shift_logits = active_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_prior_logp = prior_target_logp[..., 1:].contiguous()

    shift_mask = shift_labels != -100
    shift_safe_labels = shift_labels.clamp(min=0)

    # Per-token CE and log-prob
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logp = log_probs.gather(
        dim=-1, index=shift_safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    ce_t = -per_token_logp  # (B, T-1)

    # Mask invalid positions
    active_target_logp = per_token_logp * shift_mask.float()

    # --- Gate 1: Hardness h_t ---
    h_t = compute_hardness_gate(ce_t, shift_mask, tau_p, T_p)

    # --- Gate 2: Useful-hard q_t ---
    # Directional improvement: Δ_t⁺ = max(0, log p_θ − log p₀)
    instant_delta_plus = (active_target_logp - shift_prior_logp).clamp(min=0.0)
    instant_delta_plus = instant_delta_plus * shift_mask.float()

    q_t = compute_useful_hard_gate(instant_delta_plus, shift_mask, tau_delta, T_delta)

    # --- Final weight ---
    w_t = compute_token_weights(h_t, q_t, w_min)

    # --- Weighted CE loss ---
    weighted_ce = w_t * ce_t * shift_mask.float()
    num_valid = shift_mask.float().sum().clamp(min=1.0)
    loss = weighted_ce.sum() / num_valid

    # Return intermediates for logging
    intermediates = {
        "active_target_logp": active_target_logp,  # (B, T-1), shifted
        "instant_delta_plus": instant_delta_plus,   # (B, T-1), shifted
        "shift_mask": shift_mask,                   # (B, T-1)
        "h_t": h_t,                                 # (B, T-1)
        "q_t": q_t,                                 # (B, T-1)
        "w_t": w_t,                                 # (B, T-1)
    }

    return loss, intermediates
