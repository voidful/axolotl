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
Drift Trainer: Unified Risk Score.

Single model, single forward pass per step.
Token risk is computed from instantaneous hardness (CE) and historical
drift (EMA of CE), then mapped to reliability for weighted loss.

  z_t = β · zn(CE_t) + (1-β) · zn(d_t)
  r_t = σ(-z_t / τ)
  L   = λ · r_t · CE_t / |V|
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .drift import DriftTracker
from .loss import compute_risk_weighted_loss, compute_unified_risk

LOG = get_logger(__name__)


class AxolotlDriftTrainer(AxolotlTrainer):
    """
    Drift Trainer with unified risk scoring.

    Memory: 1× model size. No teacher, no cache, no second forward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rho = getattr(self.args, "drift_rho", 0.999) or 0.999
        self.drift_tracker = DriftTracker(rho=rho)

        LOG.info(
            "Drift trainer initialized (ρ=%.4f, β=%.2f, τ=%.2f, λ=%.2f)",
            rho,
            getattr(self.args, "drift_beta", 0.5) or 0.5,
            getattr(self.args, "drift_tau", 1.0) or 1.0,
            getattr(self.args, "drift_lambda", 1.0) or 1.0,
        )

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Single-forward unified risk score loss.

        1. Forward → logits → per-token CE
        2. Drift tracker → per-token d_t
        3. Unified risk z_t → reliability r_t
        4. Loss = λ · r_t · CE_t
        5. Update drift tracker
        """
        # Handle sample packing
        if (
            self.args.sample_packing
            and "attention_mask" in inputs
            and "position_ids" in inputs
        ):
            del inputs["attention_mask"]

        if num_items_in_batch is None and "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum().item()

        # Single forward pass
        outputs = model(**inputs)

        # Eval: standard CE
        if not model.training:
            loss = outputs.loss if outputs.loss is not None else outputs[0]
            return (loss, outputs) if return_outputs else loss

        active_logits = outputs.logits  # (B, T, V)
        labels = inputs.get("labels", None)
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        B, T = labels.shape
        valid_mask = labels != -100

        # --- Compute per-token CE (unshifted, for risk computation) ---
        with torch.no_grad():
            log_probs = F.log_softmax(active_logits, dim=-1)
            safe_labels = labels.clamp(min=0)
            token_logp = log_probs.gather(
                dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)
            ce_t = -token_logp * valid_mask.float()  # (B, T)

        # --- Get drift d_t ---
        with torch.no_grad():
            d_t = self.drift_tracker.get_drift(ce_t, valid_mask)

        # --- Compute unified risk → reliability ---
        with torch.no_grad():
            r_t = compute_unified_risk(
                ce_t=ce_t,
                d_t=d_t,
                valid_mask=valid_mask,
                beta=getattr(self.args, "drift_beta", 0.5) or 0.5,
                tau=getattr(self.args, "drift_tau", 1.0) or 1.0,
            )

        # --- Compute loss (with gradient) ---
        loss, per_token_ce_shifted = compute_risk_weighted_loss(
            active_logits=active_logits,
            labels=labels,
            r_t=r_t,
            lam=getattr(self.args, "drift_lambda", 1.0) or 1.0,
            num_items_in_batch=num_items_in_batch,
        )

        # --- Update drift tracker ---
        self.drift_tracker.step(ce_t, valid_mask)

        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Standard CE for eval."""
        model_inputs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "position_ids", "labels")
        }

        with torch.no_grad():
            model.eval()
            try:
                outputs = model(**model_inputs)
            finally:
                model.train()

        loss = outputs.loss if outputs.loss is not None else outputs[0]

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs.get("labels", None)
        return (
            loss.detach(),
            logits.detach() if logits is not None else None,
            labels.detach() if labels is not None else None,
        )
