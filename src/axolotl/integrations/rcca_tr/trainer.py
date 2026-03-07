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
RCCA-TR Trainer.

Implements the Reliability-Calibrated Conflict-Aware Trust-Region Fine-Tuning
trainer as an AxolotlTrainer subclass.

The trainer manages three models:
  1. Active model (p_θ) — the model being trained.
  2. Frozen model (p_0) — preserves the original prior, never updated.
  3. EMA model (p_ema) — slow-moving evidence accumulator.

On each step, it computes:
  - Conflict score α_t: does the label contradict the prior?
  - Reliability score r_t: is the prior stable and not outdated?
  - Trust-region loss: α_t · CE + λ · g(r_t) · KL(p_θ ∥ p_0)
"""

import torch
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .ema import create_ema_model, create_frozen_model
from .loss import (
    compute_conflict_score,
    compute_evidence_drift,
    compute_reliability,
    compute_stability,
    compute_trust_region_loss,
)

LOG = get_logger(__name__)


class AxolotlRCCATRTrainer(AxolotlTrainer):
    """
    Custom trainer for Reliability-Calibrated Conflict-Aware Trust-Region Fine-Tuning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        LOG.info("Initializing RCCA-TR trainer...")

        # Create frozen base model (p_0)
        base_model = self.model
        if hasattr(base_model, "module"):
            base_model = base_model.module

        LOG.info("Creating frozen base model (p_0)...")
        self.frozen_model = create_frozen_model(base_model)

        LOG.info("Creating EMA model (p_ema)...")
        self.ema_model = create_ema_model(base_model)

        # Stability cache
        self._stability_cache = None
        self._stability_cache_step = -1

        LOG.info("RCCA-TR trainer initialized with frozen model and EMA model.")

    def _get_frozen_logits(self, inputs):
        """Forward pass through frozen model (no grad)."""
        with torch.no_grad():
            frozen_inputs = {
                k: v for k, v in inputs.items()
                if k in ("input_ids", "attention_mask", "position_ids")
            }
            frozen_outputs = self.frozen_model(**frozen_inputs)
            return frozen_outputs.logits

    def _get_ema_logits(self, inputs):
        """Forward pass through EMA model (no grad)."""
        with torch.no_grad():
            ema_inputs = {
                k: v for k, v in inputs.items()
                if k in ("input_ids", "attention_mask", "position_ids")
            }
            ema_outputs = self.ema_model(**ema_inputs)
            return ema_outputs.logits

    def _compute_stability_with_perturbations(
        self, inputs, frozen_logits_ref, num_perturbations
    ):
        """
        Compute stability by doing K forward passes through the frozen model
        with dropout enabled (perturbation via dropout noise).

        This is cached and only recomputed every N steps.
        """
        current_step = self.state.global_step if self.state else 0
        update_interval = getattr(
            self.args, "rcca_tr_stability_update_interval", 50
        )

        # Use cache if available and recent
        if (
            self._stability_cache is not None
            and (current_step - self._stability_cache_step) < update_interval
        ):
            # Resize cache if batch size changed
            batch_size, seq_len = frozen_logits_ref.shape[:2]
            cache_b, cache_t = self._stability_cache.shape
            if cache_b == batch_size and cache_t == seq_len:
                return self._stability_cache
            # Fall through to recompute if shapes differ

        perturbation_logits = []
        with torch.no_grad():
            pert_inputs = {
                k: v for k, v in inputs.items()
                if k in ("input_ids", "attention_mask", "position_ids")
            }
            # Temporarily enable dropout for perturbation
            self.frozen_model.train()  # enables dropout
            for _ in range(num_perturbations):
                pert_outputs = self.frozen_model(**pert_inputs)
                perturbation_logits.append(pert_outputs.logits)
            self.frozen_model.eval()  # disable dropout again

        stability = compute_stability(perturbation_logits, frozen_logits_ref)

        # Cache the result
        self._stability_cache = stability
        self._stability_cache_step = current_step

        return stability

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute the RCCA-TR trust-region loss.

        Steps:
            1. Forward pass through active model → p_θ
            2. Forward pass through frozen model → p_0
            3. Forward pass through EMA model → p_ema
            4. Compute conflict score α_t
            5. Compute reliability score r_t (stability + evidence)
            6. Compute trust-region loss
        """
        # Handle sample packing
        if (
            self.args.sample_packing
            and hasattr(inputs, "attention_mask")
            and hasattr(inputs, "position_ids")
        ):
            del inputs["attention_mask"]

        if num_items_in_batch is None and "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum().item()

        # 1. Forward pass through active model
        outputs = model(**inputs)
        active_logits = outputs.logits  # (B, T, V)

        labels = inputs.get("labels", None)
        if labels is None:
            # Fall back to standard loss if no labels
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        # 2. Forward pass through frozen model
        frozen_logits = self._get_frozen_logits(inputs)

        # 3. Forward pass through EMA model
        ema_logits = self._get_ema_logits(inputs)

        # 4. Compute conflict score α_t
        alpha_t = compute_conflict_score(
            frozen_logits=frozen_logits,
            labels=labels,
            lambda1=getattr(self.args, "rcca_tr_conflict_lambda1", 1.0) or 1.0,
            lambda2=getattr(self.args, "rcca_tr_conflict_lambda2", 0.5) or 0.5,
            tau=getattr(self.args, "rcca_tr_conflict_tau", 1.0) or 1.0,
        )

        # 5. Compute reliability score r_t
        # 5a. Stability (with caching)
        num_perturbations = getattr(self.args, "rcca_tr_num_perturbations", 3) or 3
        stability = self._compute_stability_with_perturbations(
            inputs, frozen_logits, num_perturbations
        )

        # 5b. Evidence drift
        evidence_reliability = compute_evidence_drift(
            ema_logits=ema_logits,
            frozen_logits=frozen_logits,
        )

        # 5c. Combined reliability
        r_t = compute_reliability(
            stability=stability,
            evidence_reliability=evidence_reliability,
            beta=getattr(self.args, "rcca_tr_reliability_beta", 0.5) or 0.5,
            tau=getattr(self.args, "rcca_tr_reliability_tau", 1.0) or 1.0,
        )

        # 6. Compute trust-region loss
        loss = compute_trust_region_loss(
            active_logits=active_logits,
            frozen_logits=frozen_logits,
            labels=labels,
            alpha_t=alpha_t,
            r_t=r_t,
            kl_lambda=getattr(self.args, "rcca_tr_kl_lambda", 1.0) or 1.0,
            epsilon_min=getattr(self.args, "rcca_tr_epsilon_min", 0.01) or 0.01,
            epsilon_max=getattr(self.args, "rcca_tr_epsilon_max", 1.0) or 1.0,
            use_smooth=getattr(self.args, "rcca_tr_use_smooth_objective", True),
            num_items_in_batch=num_items_in_batch,
        )

        return (loss, outputs) if return_outputs else loss
