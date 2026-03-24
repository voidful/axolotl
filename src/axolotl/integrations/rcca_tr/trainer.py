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
  - Self-paced score s_t from active model confidence
  - Drift-based reliability r_t (no EMA forward)
  - Trust-region loss: α_t · s_t · CE + λ · g(r_t) · KL_proxy
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
    compute_self_paced_score,
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

        # Initialize drift buffer (replaces EMA model)
        ema_decay = getattr(self.args, "rcca_tr_ema_decay", 0.999) or 0.999
        drift_gamma = getattr(self.args, "rcca_tr_drift_gamma", 1.0) or 1.0
        self.drift_buffer = DriftBuffer(decay=ema_decay, gamma=drift_gamma)

        LOG.info(
            "RCCA-TR A+ trainer initialized (drift_decay=%.4f, drift_gamma=%.2f)",
            ema_decay,
            drift_gamma,
        )

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self._signature_columns:
            for col in ["prior_target_logp", "prior_margin"]:
                if col not in self._signature_columns:
                    self._signature_columns.append(col)

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

        # Pop RCCA-TR fields before model forward (model doesn't accept them)
        prior_target_logp = inputs.pop("prior_target_logp", None)
        prior_margin = inputs.pop("prior_margin", None)

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

        # Fallback: zeros if prior values not in batch
        if prior_target_logp is None or prior_margin is None:
            LOG.warning(
                "Prior cache values missing from batch. "
                "Falling back to zero tensors — RCCA-TR will degrade into weighted CE."
            )
            prior_target_logp = torch.zeros(B, T, device=labels.device)
            prior_margin = torch.zeros(B, T, device=labels.device)

        # 1. Compute conflict score α_t from cached values
        alpha_t = compute_conflict_score_from_cache(
            prior_target_logp=prior_target_logp,
            prior_margin=prior_margin,
            valid_mask=valid_mask,
            lambda1=getattr(self.args, "rcca_tr_conflict_lambda1", 1.0) or 1.0,
            lambda2=getattr(self.args, "rcca_tr_conflict_lambda2", 0.5) or 0.5,
            tau=getattr(self.args, "rcca_tr_conflict_tau", 1.0) or 1.0,
        )

        # 2. Compute drift-based reliability using previous running average
        # Get active model's log p(y_t) for drift estimation
        # CRITICAL: active_logits[b, t] predicts token at position t+1 (next-token).
        # To get log p_θ(y_t | x, y_{<t}), we need logits at position t-1 aligned
        # with label at position t. This matches prior_target_logp's convention.
        with torch.no_grad():
            shift_active_log_probs = F.log_softmax(active_logits[..., :-1, :], dim=-1)  # (B, T-1, V)
            shift_labels_for_gather = labels[..., 1:].clamp(min=0)  # (B, T-1)
            shift_valid = (labels[..., 1:] != -100)
            active_target_logp_shifted = shift_active_log_probs.gather(
                dim=-1, index=shift_labels_for_gather.unsqueeze(-1)
            ).squeeze(-1)  # (B, T-1)
            active_target_logp_shifted = active_target_logp_shifted * shift_valid.float()
            # Pad position 0 with 0.0 to restore shape (B, T), matching prior_target_logp
            active_target_logp = F.pad(active_target_logp_shifted, (1, 0), value=0.0)
            # Build matching valid mask: position 0 is ALWAYS invalid (no next-token prediction exists)
            shifted_valid_mask = F.pad(shift_valid, (1, 0), value=False)

        drift = self.drift_buffer.get_current_drift(
            active_target_logp=active_target_logp,
            prior_target_logp=prior_target_logp,
            valid_mask=shifted_valid_mask,
        )

        r_t_prior = compute_reliability_from_drift(
            drift=drift,
            valid_mask=shifted_valid_mask,
            gamma=getattr(self.args, "rcca_tr_drift_gamma", 1.0) or 1.0,
            tau=getattr(self.args, "rcca_tr_reliability_tau", 1.0) or 1.0,
        )

        s_t = compute_self_paced_score(
            active_target_logp=active_target_logp,
            valid_mask=shifted_valid_mask,
            tau=getattr(self.args, "rcca_tr_self_tau", 1.0) or 1.0,
        )

        # 3. Compute trust-region loss with dual-branch control
        # r_t_prior dictates trust-region strength (KL). s_t dictates self-paced learning speed (CE).
        loss, _ = compute_trust_region_loss_cached(
            active_logits=active_logits,
            labels=labels,
            alpha_t=alpha_t,
            r_t=r_t_prior,
            prior_target_logp=prior_target_logp,
            s_t=s_t,
            kl_lambda=getattr(self.args, "rcca_tr_kl_lambda", 1.0) or 1.0,
            epsilon_min=getattr(self.args, "rcca_tr_epsilon_min", 0.01) or 0.01,
            epsilon_max=getattr(self.args, "rcca_tr_epsilon_max", 1.0) or 1.0,
            use_smooth=getattr(self.args, "rcca_tr_use_smooth_objective", True),
        )

        # 4. Update drift buffer with this step's values
        self.drift_buffer.step(
            active_target_logp=active_target_logp,
            prior_target_logp=prior_target_logp,
            valid_mask=shifted_valid_mask,
        )

        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to use standard CE loss during eval.

        The default Trainer.prediction_step calls compute_loss, but under
        DeepSpeed ZeRO-2, model.training may still be True during eval.
        This override explicitly puts the model in eval mode and computes
        CE loss directly, ensuring eval_loss is always populated.
        """
        # Filter inputs to only what the model accepts
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

        # Return loss + logits + labels for metric computation
        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs.get("labels", None)
        return (
            loss.detach(),
            logits.detach() if logits is not None else None,
            labels.detach() if labels is not None else None,
        )
