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
Drift buffer for RCCA-TR.

Tracks how much the active model's predictions diverge from the
historical running mean over time.  Provides a temporal evidence
signal without storing a full model copy.

    drift_t = log p_θ(y_t) − running_mean
    running_mean ← decay · running_mean + (1 − decay) · batch_mean
"""

import torch


class DriftBuffer:
    """
    Minimal, paper-grade drift buffer.

    Maintains a scalar EMA of per-token log-probabilities.
    Drift is defined as the deviation of the current token's
    log-prob from this running mean.

    This is the *only* temporal state the method keeps — no frozen
    model, no prior cache, no per-token history.
    """

    def __init__(self, decay: float = 0.999):
        """
        Args:
            decay: EMA decay for the running mean. Higher = slower adaptation.
        """
        self.decay = decay
        self.running_mean: float = 0.0

    def step(
        self,
        active_target_logp: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        Update the running mean with the current batch.

        Args:
            active_target_logp: log p_θ(y_t), shape (B, T).
            valid_mask: Boolean mask for valid (non-padding) tokens, shape (B, T).
        """
        if valid_mask.any():
            batch_mean = active_target_logp[valid_mask].mean().item()
        else:
            batch_mean = 0.0

        self.running_mean = (
            self.decay * self.running_mean
            + (1.0 - self.decay) * batch_mean
        )

    def get_current_drift(
        self,
        active_target_logp: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token drift = current log-prob − running mean.

        Positive drift → token is easier than average (model confident).
        Negative drift → token is harder than average (model struggling).

        Args:
            active_target_logp: log p_θ(y_t), shape (B, T).
            valid_mask: Boolean mask, shape (B, T).

        Returns:
            Per-token drift, shape (B, T). Masked positions are 0.
        """
        drift = active_target_logp - self.running_mean
        return drift * valid_mask.float()

    @property
    def state(self) -> float:
        """Current running mean (for logging)."""
        return self.running_mean
