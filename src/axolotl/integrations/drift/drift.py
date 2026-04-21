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
Minimal EMA reference buffer for drift-loss.

The no-cache default keeps a single scalar EMA over valid target log-probabilities:

  m <- decay * m + (1 - decay) * mean_valid(log p_theta(y_t))

This is the only persistent temporal state needed by drift-loss-EMA.
"""

from __future__ import annotations

import torch


class DriftMeanBuffer:
    """
    Maintain a scalar EMA reference over valid target log-probabilities.
    """

    def __init__(self, decay: float = 0.999):
        """
        Args:
            decay: EMA decay. Higher values keep a longer temporal memory.
        """
        self.decay = decay
        self.running_mean: float = 0.0
        self.initialized: bool = False

    def step(
        self,
        active_target_logp: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        Update the EMA reference with the current batch mean.

        Args:
            active_target_logp: Valid-token log-probs, shape (B, T).
            valid_mask: Boolean mask for valid tokens, shape (B, T).
        """
        if valid_mask.any():
            batch_mean = active_target_logp[valid_mask].mean().item()
        else:
            batch_mean = 0.0

        if not self.initialized:
            self.running_mean = batch_mean
            self.initialized = True
            return

        self.running_mean = (
            self.decay * self.running_mean
            + (1.0 - self.decay) * batch_mean
        )

    def get_reference(self) -> float:
        """
        Return the current scalar EMA reference.
        """
        return float(self.running_mean)

    @property
    def state(self) -> float:
        """Alias for logging consistency with other buffers."""
        return self.running_mean
