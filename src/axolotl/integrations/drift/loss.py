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
Core loss functions for drift-loss.

Paper default:
  1. CE already captures hardness.
  2. Drift only captures usefulness/improvement versus a reference state.
  3. Gamma is the only focal-style sharpening parameter.

Two reference modes are supported:
  - scalar EMA reference: current log-prob versus a running mean
  - token-wise prior reference: current log-prob versus token-aligned prior log-prob
"""

from __future__ import annotations

import torch


def _shift_for_next_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shift logits and labels for next-token prediction.

    Args:
        logits: Model logits, shape (B, T, V).
        labels: Token labels, shape (B, T).

    Returns:
        Tuple of shifted logits, shifted labels, and valid-token mask.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = shift_labels != -100
    return shift_logits, shift_labels, shift_mask


def _gather_target_logp(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Gather target token log-probabilities without materializing full log_softmax.

    Args:
        shift_logits: Shifted logits, shape (B, T-1, V).
        shift_labels: Shifted labels, shape (B, T-1).

    Returns:
        Target log-probs, shape (B, T-1).
    """
    safe_labels = shift_labels.clamp(min=0)
    target_logit = shift_logits.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    log_z = torch.logsumexp(shift_logits, dim=-1)
    return target_logit - log_z


def _broadcast_reference(
    reference_target_logp: torch.Tensor | float,
    target_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, str]:
    """
    Broadcast a scalar or token-wise reference to the shifted target shape.

    Args:
        reference_target_logp: Scalar EMA reference or token-wise prior log-probs.
        target_shape: Expected shifted token shape, usually (B, T-1).
        device: Output device.
        dtype: Output dtype.

    Returns:
        Tuple of broadcast reference log-probs and reference type.
    """
    if not torch.is_tensor(reference_target_logp):
        ref = torch.full(
            target_shape,
            float(reference_target_logp),
            device=device,
            dtype=dtype,
        )
        return ref, "scalar"

    ref = reference_target_logp.to(device=device, dtype=dtype)

    if ref.ndim == 0:
        ref = torch.full(
            target_shape,
            float(ref.item()),
            device=device,
            dtype=dtype,
        )
        return ref, "scalar"

    if ref.shape == target_shape:
        return ref, "token"

    if (
        ref.ndim == 2
        and ref.shape[0] == target_shape[0]
        and ref.shape[1] == target_shape[1] + 1
    ):
        return ref[..., 1:].contiguous(), "token"

    raise ValueError(
        f"Unsupported reference_target_logp shape {tuple(ref.shape)} "
        f"for target shape {tuple(target_shape)}"
    )


def compute_drift_focal_loss(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    reference_target_logp: torch.Tensor | float,
    gamma: float = 2.0,
    eps: float = 1e-6,
    detach_weights: bool = True,
    num_items_in_batch: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """
    Compute the one-parameter drift-loss objective.

    Args:
        active_logits: Logits from the active model, shape (B, T, V).
        labels: Ground-truth token IDs, shape (B, T), with -100 ignored.
        reference_target_logp: Scalar EMA reference or token-wise prior log-prob.
        gamma: Focal-style sharpening parameter. gamma=0 recovers standard CE.
        eps: Numerical stability term.
        detach_weights: If True, detach the weight path (paper default).
        num_items_in_batch: Optional packed-sequence reduction denominator.

    Returns:
        Tuple of scalar loss and intermediate stats.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    shift_logits, shift_labels, shift_mask = _shift_for_next_token(
        active_logits,
        labels,
    )
    mask_float = shift_mask.float()

    active_target_logp = _gather_target_logp(shift_logits, shift_labels)
    active_target_logp = active_target_logp * mask_float
    ce_t = -active_target_logp

    ref_logp, ref_type = _broadcast_reference(
        reference_target_logp=reference_target_logp,
        target_shape=active_target_logp.shape,
        device=active_target_logp.device,
        dtype=active_target_logp.dtype,
    )
    ref_logp = ref_logp * mask_float

    delta_t = (active_target_logp - ref_logp) * mask_float

    if shift_mask.any():
        valid_delta = delta_t[shift_mask]
        std_delta = valid_delta.detach().std(unbiased=False)
        if ref_type == "scalar":
            mu_delta = delta_t.new_zeros(())
            delta_norm = delta_t / (std_delta + eps)
        else:
            mu_delta = valid_delta.detach().mean()
            delta_norm = (delta_t - mu_delta) / (std_delta + eps)
    else:
        mu_delta = delta_t.new_zeros(())
        std_delta = delta_t.new_ones(())
        delta_norm = delta_t

    delta_norm = delta_norm * mask_float
    u_t = torch.sigmoid(delta_norm) * mask_float

    if gamma == 0.0:
        w_raw = mask_float.clone()
    else:
        w_raw = u_t.clamp(min=eps).pow(gamma) * mask_float

    if shift_mask.any():
        mean_w = w_raw[shift_mask].detach().mean()
    else:
        mean_w = w_raw.new_ones(())

    w_t = (w_raw / (mean_w + eps)) * mask_float
    weighted_w = w_t.detach() if detach_weights else w_t
    weighted_ce = weighted_w * ce_t * mask_float

    if num_items_in_batch is not None:
        denom = active_logits.new_tensor(max(float(num_items_in_batch), 1.0))
    else:
        denom = mask_float.sum().clamp(min=1.0)

    loss = weighted_ce.sum() / denom

    stats: dict[str, object] = {
        "shift_mask": shift_mask,
        "reference_type": ref_type,
        "active_target_logp": active_target_logp.detach(),
        "reference_target_logp": ref_logp.detach(),
        "ce_t": ce_t.detach(),
        "delta_t": delta_t.detach(),
        "delta_norm": delta_norm.detach(),
        "u_t": u_t.detach(),
        "w_raw": w_raw.detach(),
        "w_t": w_t.detach(),
        "mu_delta": mu_delta.detach(),
        "std_delta": std_delta.detach(),
        "mean_w": mean_w.detach(),
    }

    return loss, stats
