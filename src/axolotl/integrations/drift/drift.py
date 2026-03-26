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
EMA drift tracker for Drift.

Maintains a running exponential moving average of per-token CE:
    d_t = ρ · d_{t-1} + (1 - ρ) · CE_t

This provides the "historical difficulty" signal for the unified risk score.
"""

import torch


class DriftTracker:
    """
    Lightweight EMA tracker for token-level cross-entropy drift.

    Stores a single scalar running average (not a full model copy).
    On each step, blends the batch mean CE into the running average.
    At query time, returns per-token drift estimates by blending
    the instant CE with the running average.
    """

    def __init__(self, rho: float = 0.999):
        """
        Args:
            rho: EMA decay. Higher = slower adaptation to new data.
        """
        self.rho = rho
        self._running_ce: float = 0.0
        self._initialized: bool = False

    def get_drift(
        self,
        ce_t: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token drift: d_t = ρ · d_running + (1-ρ) · CE_t.

        On the first call (before any step()), d_running=0, so drift ≈ CE_t.

        Args:
            ce_t: Per-token CE, shape (B, T).
            valid_mask: Boolean mask, shape (B, T).

        Returns:
            Per-token drift values, shape (B, T).
        """
        d_t = self.rho * self._running_ce + (1.0 - self.rho) * ce_t
        return d_t * valid_mask.float()

    def step(self, ce_t: torch.Tensor, valid_mask: torch.Tensor):
        """
        Update the running CE average with this batch's mean.

        Args:
            ce_t: Per-token CE, shape (B, T).
            valid_mask: Boolean mask, shape (B, T).
        """
        if valid_mask.any():
            batch_mean = ce_t[valid_mask].mean().item()
        else:
            batch_mean = 0.0

        if not self._initialized:
            self._running_ce = batch_mean
            self._initialized = True
        else:
            self._running_ce = (
                self.rho * self._running_ce
                + (1.0 - self.rho) * batch_mean
            )

    @property
    def running_ce(self) -> float:
        """Current running CE average."""
        return self._running_ce
