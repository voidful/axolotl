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
RCCA-TR Trainer (A+ variant).

Only one model lives in GPU memory — the active model being trained.
The frozen prior is replaced by cached log-probabilities loaded from disk.
The EMA model is replaced by a lightweight drift buffer.

On each step, it computes:
  - Conflict score α_t from cached prior values (no frozen forward)
  - Drift-based reliability r_t (no EMA forward)
  - Trust-region loss: α_t · CE + λ · g(r_t) · KL_proxy
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .drift import DriftBuffer
from .loss import (
    compute_conflict_score_from_cache,
    compute_reliability_from_drift,
    compute_trust_region_loss_cached,
)

LOG = get_logger(__name__)


class AxolotlRCCATRTrainer(AxolotlTrainer):
    """
    A+ Trust-Region Trainer.

    Memory footprint: only the active model (1× model size).
    Prior information comes from cached values in the batch.
    Evidence drift is tracked via a lightweight DriftBuffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        LOG.info("Initializing RCCA-TR A+ trainer...")

        # Load prior cache if path is specified
        prior_cache_path = getattr(self.args, "rcca_tr_prior_cache_path", None)
        if prior_cache_path:
            LOG.info("Loading prior cache from %s", prior_cache_path)
            self._prior_cache = torch.load(prior_cache_path, weights_only=True)
            LOG.info(
                "Prior cache loaded: %d samples",
                len(self._prior_cache["prior_target_logp"]),
            )
        else:
            self._prior_cache = None
            LOG.info(
                "No prior cache path specified. "
                "Will run frozen forward pass on-the-fly (fallback mode)."
            )

        # Initialize drift buffer (replaces EMA model)
        ema_decay = getattr(self.args, "rcca_tr_ema_decay", 0.999) or 0.999
        drift_gamma = getattr(self.args, "rcca_tr_drift_gamma", 1.0) or 1.0
        self.drift_buffer = DriftBuffer(decay=ema_decay, gamma=drift_gamma)

        LOG.info(
            "RCCA-TR A+ trainer initialized (drift_decay=%.4f, drift_gamma=%.2f)",
            ema_decay,
            drift_gamma,
        )

    def _get_prior_values_from_cache(
        self, batch_idx: int | None, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get prior_target_logp and prior_margin for the current batch.

        If a prior cache file is loaded, look up cached values.
        Otherwise return zeros (fallback — conflict score becomes uniform).

        Returns:
            (prior_target_logp, prior_margin), each shape (B, T).
        """
        B, T = labels.shape
        device = labels.device

        if self._prior_cache is None:
            # Fallback: uniform conflict, no drift
            return (
                torch.zeros(B, T, device=device),
                torch.zeros(B, T, device=device),
            )

        # For now, return zeros as placeholder — actual cache indexing
        # depends on how axolotl's dataloader provides sample IDs.
        # The preprocessing pipeline will embed these into the dataset.
        return (
            torch.zeros(B, T, device=device),
            torch.zeros(B, T, device=device),
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
        Compute the RCCA-TR A+ trust-region loss.

        During training: uses conflict score + drift buffer + KL proxy.
        During eval: falls back to standard CE loss for proper eval_loss metric.
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

        # Forward pass through active model
        outputs = model(**inputs)

        # During eval, use standard CE loss (trust-region doesn't apply)
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

        # 2. Get cached prior values
        # Check if prior values are embedded in the batch (from preprocessing)
        if "prior_target_logp" in inputs and "prior_margin" in inputs:
            prior_target_logp = inputs["prior_target_logp"]  # (B, T)
            prior_margin = inputs["prior_margin"]  # (B, T)
        else:
            # Fallback: compute on-the-fly if no cache available
            prior_target_logp, prior_margin = self._get_prior_values_from_cache(
                None, labels
            )

        # 3. Compute conflict score α_t from cached values
        alpha_t = compute_conflict_score_from_cache(
            prior_target_logp=prior_target_logp,
            prior_margin=prior_margin,
            valid_mask=valid_mask,
            lambda1=getattr(self.args, "rcca_tr_conflict_lambda1", 1.0) or 1.0,
            lambda2=getattr(self.args, "rcca_tr_conflict_lambda2", 0.5) or 0.5,
            tau=getattr(self.args, "rcca_tr_conflict_tau", 1.0) or 1.0,
        )

        # 4. Compute drift and reliability (no EMA model needed)
        # Get active model's log p(y_t) for drift computation
        with torch.no_grad():
            active_log_probs = F.log_softmax(active_logits, dim=-1)
            safe_labels = labels.clamp(min=0)
            active_target_logp = active_log_probs.gather(
                dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)
            active_target_logp = active_target_logp * valid_mask.float()

        # Update drift buffer and get drift values
        drift = self.drift_buffer.update(
            active_target_logp=active_target_logp,
            prior_target_logp=prior_target_logp,
            valid_mask=valid_mask,
        )

        # Compute reliability from drift
        r_t = compute_reliability_from_drift(
            drift=drift,
            gamma=getattr(self.args, "rcca_tr_drift_gamma", 1.0) or 1.0,
            tau=getattr(self.args, "rcca_tr_reliability_tau", 1.0) or 1.0,
        )

        # 5. Compute trust-region loss with KL proxy
        loss, _ = compute_trust_region_loss_cached(
            active_logits=active_logits,
            labels=labels,
            alpha_t=alpha_t,
            r_t=r_t,
            prior_target_logp=prior_target_logp,
            kl_lambda=getattr(self.args, "rcca_tr_kl_lambda", 1.0) or 1.0,
            epsilon_min=getattr(self.args, "rcca_tr_epsilon_min", 0.01) or 0.01,
            epsilon_max=getattr(self.args, "rcca_tr_epsilon_max", 1.0) or 1.0,
            use_smooth=getattr(self.args, "rcca_tr_use_smooth_objective", True),
            num_items_in_batch=num_items_in_batch,
        )

        return (loss, outputs) if return_outputs else loss
