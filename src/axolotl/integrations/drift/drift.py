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
Drift buffer for Drift variant.

Tracks how much the active model's predictions diverge from baseline (zeros)
over time, providing a temporal evidence signal without storing a full model copy.

Usage:
    drift_t = decay * drift_t + (1-decay) * |log p_θ(y_t) - 0|
    r_evi = exp(-gamma * drift_t)
"""

import torch


class DriftBuffer:
    """
    Maintains a running exponential drift statistic.

    Instead of keeping an entire EMA model, we track a single scalar drift
    per token position in each batch. This provides the temporal evidence
    signal for reliability estimation.

    Without a prior cache, drift equals the active model's CE (|log p_θ(y_t)|).
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
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token drift using the current running average (without updating it).

        drift_t = |log p_θ(y_t)|  (= CE_t since log p is negative)

        Returns blended drift: 0.5 * instant + 0.5 * running_average.

        Args:
            active_target_logp: log p_θ(y_t), shape (B, T).
            valid_mask: Boolean mask for valid tokens, shape (B, T).

        Returns:
            Per-token drift values, shape (B, T).
        """
        instant_drift = active_target_logp.abs()
        blended_drift = (
            0.5 * instant_drift
            + 0.5 * self._running_drift
        )
        return blended_drift * valid_mask.float()

    def step(
        self,
        active_target_logp: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """
        Update the running drift average with the current batch's mean drift.

        d = decay * d + (1 - decay) * mean(|log p_θ(y_t)|)

        Args:
            active_target_logp: log p_θ(y_t), shape (B, T).
            valid_mask: Boolean mask for valid tokens, shape (B, T).
        """
        instant_drift = active_target_logp.abs()
        if valid_mask.any():
            batch_mean_drift = instant_drift[valid_mask].mean().item()
        else:
            batch_mean_drift = 0.0

        self._running_drift = (
            self.decay * self._running_drift
            + (1.0 - self.decay) * batch_mean_drift
        )

    def get_evidence_reliability(self, drift: torch.Tensor) -> torch.Tensor:
        """
        Convert drift values to evidence reliability scores.

        r_evi = exp(-gamma * drift)

        High drift → low reliability (model uncertain).
        Low drift → high reliability (model confident).

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
