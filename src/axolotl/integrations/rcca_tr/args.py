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
Plugin args for Drift-Trust.

A token-level regularization family built on a shared drift signal
and two transfer functions:

  - ce:            Standard cross-entropy (baseline).
  - hardness:      Self-paced hardness weighting only (ablation).
  - drift_trust_s: Suppressive mapping — best for noisy alignment.
  - drift_trust_a: Anchoring mapping — best for clean domain specialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel


class RCCATRArgs(BaseModel):
    """
    Input args for Drift-Trust fine-tuning.
    """

    rcca_tr_trainer: Optional[bool] = None

    # --- Method mode ---
    rcca_tr_mode: Optional[Literal[
        "ce", "hardness", "drift_trust_s", "drift_trust_a",
        # Legacy aliases (backward compat with existing configs)
        "drift_only", "drift",
    ]] = "drift_trust_s"

    # --- Shared drift buffer ---
    rcca_tr_ema_decay: Optional[float] = 0.999  # EMA decay for running mean

    # --- Drift-Trust-S (suppressive) hyperparameters ---
    rcca_tr_self_tau: Optional[float] = 1.0    # self-paced score temperature
    rcca_tr_drift_tau: Optional[float] = 1.0   # drift score temperature
    rcca_tr_w_min: Optional[float] = 0.05      # minimum token weight floor
    rcca_tr_beta: Optional[float] = 0.5        # balance: β·s_t + (1-β)·r_t

    # --- Drift-Trust-A (anchoring) hyperparameters ---
    rcca_tr_drift_gamma: Optional[float] = 1.0   # |drift| decay rate
    rcca_tr_anchor_base: Optional[float] = 0.1    # base weight w_0
    rcca_tr_anchor_lambda: Optional[float] = 4.0  # amplification factor λ
    rcca_tr_reliability_tau: Optional[float] = 1.0 # sigmoid temperature for r_t


@dataclass
class RCCATRTrainingArgsMixin:
    """
    Additional training args injected into HuggingFace TrainingArguments.
    """

    rcca_tr_mode: Optional[str] = "drift_trust_s"
    rcca_tr_ema_decay: Optional[float] = 0.999
    rcca_tr_self_tau: Optional[float] = 1.0
    rcca_tr_drift_tau: Optional[float] = 1.0
    rcca_tr_w_min: Optional[float] = 0.05
    rcca_tr_beta: Optional[float] = 0.5
    rcca_tr_drift_gamma: Optional[float] = 1.0
    rcca_tr_anchor_base: Optional[float] = 0.1
    rcca_tr_anchor_lambda: Optional[float] = 4.0
    rcca_tr_reliability_tau: Optional[float] = 1.0
