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
Drift buffer for RCCA-TR A+ variant.

Replaces the full EMA model with a lightweight running drift statistic.
The drift tracks how much the active model's predictions diverge from the
frozen prior over time, providing temporal evidence signal without storing
a full model copy.

Usage:
    drift_t = decay * drift_t + (1-decay) * |log p_θ(y_t) - log p_0(y_t)|
    r_evi = exp(-gamma * drift_t)
"""

import torch


class DriftBuffer:
    """
    Maintains a running exponential drift statistic.

    Instead of keeping an entire EMA model (~9B params), we track a single
    scalar drift per token position in each batch. This provides the temporal
    evidence signal for reliability estimation.

    The drift measures how much the active model has diverged from the frozen
    prior's predictions over the course of training.
    """

    def __init__(self, decay: float = 0.999, gamma: float = 1.0):
        """
        Args:
            decay: EMA decay for the drift buffer. Higher = slower adaptation.
            gamma: Scaling factor for drift → reliability mapping.
        """
        self.decay = decay
        self.gamma = gamma
        self._running_drift = 0.0  # scalar running average

    def get_current_drift(
        self,
        active_target_logp: torch.Tensor,
        prior_target_logp: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token drift using the current running average (without updating it).

        drift_t = |log p_θ(y_t) - log p_0(y_t)|

        Returns blended drift: 0.5 * instant + 0.5 * running_average.

        Args:
            active_target_logp: log p_θ(y_t), shape (B, T).
            prior_target_logp: log p_0(y_t), shape (B, T).
            valid_mask: Boolean mask for valid tokens, shape (B, T).

        Returns:
            Per-token drift values, shape (B, T).
        """
        instant_drift = (active_target_logp - prior_target_logp).abs()
        blended_drift = (
            0.5 * instant_drift
            + 0.5 * self._running_drift
        )
        return blended_drift * valid_mask.float()

    def step(
        self,
        active_target_logp: torch.Tensor,
        prior_target_logp: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """
        Update the running drift average with the current batch's mean drift.

        d = decay * d + (1 - decay) * mean(|log p_θ(y_t) - log p_0(y_t)|)

        Args:
            active_target_logp: log p_θ(y_t), shape (B, T).
            prior_target_logp: log p_0(y_t), shape (B, T).
            valid_mask: Boolean mask for valid tokens, shape (B, T).
        """
        instant_drift = (active_target_logp - prior_target_logp).abs()
        if valid_mask.any():
            batch_mean_drift = instant_drift[valid_mask].mean().item()
        else:
            batch_mean_drift = 0.0

        self._running_drift = (
            self.decay * self._running_drift
            + (1.0 - self.decay) * batch_mean_drift
        )

    def update(
        self,
        active_target_logp: torch.Tensor,
        prior_target_logp: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update drift buffer and return current per-token drift values.

        Convenience method that calls get_current_drift() then step().

        Args:
            active_target_logp: log p_θ(y_t), shape (B, T).
            prior_target_logp: log p_0(y_t), shape (B, T).
            valid_mask: Boolean mask for valid tokens, shape (B, T).

        Returns:
            Per-token drift values (blended), shape (B, T).
        """
        drift = self.get_current_drift(active_target_logp, prior_target_logp, valid_mask)
        self.step(active_target_logp, prior_target_logp, valid_mask)
        return drift

    def get_evidence_reliability(self, drift: torch.Tensor) -> torch.Tensor:
        """
        Convert drift values to evidence reliability scores.

        r_evi = exp(-gamma * drift)

        High drift → low reliability (prior is outdated).
        Low drift → high reliability (prior still accurate).

        Args:
            drift: Per-token drift values, shape (B, T).

        Returns:
            Evidence reliability scores in [0, 1], shape (B, T).
        """
        return torch.exp(-self.gamma * drift)

    @property
    def running_drift(self) -> float:
        """Current running drift average."""
        return self._running_drift
