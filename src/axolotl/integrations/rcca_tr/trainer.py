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
RCCA-TR Trainer: Suppress-by-Default, Rescue-if-Useful.

Only one model lives in GPU memory — the active model being trained.
The frozen prior is replaced by cached log-probabilities loaded from disk.

Two orthogonal gates per token:
  A. Hardness gate h_t  — is this token hard enough to warrant suppression?
  B. Useful-hard gate q_t — does this hard token improve over the prior?

Final weight: w_t = w_min + (1-w_min) · (1 - h_t·(1-q_t))
Loss: L = Σ w_t · CE_t / |V|
"""

import torch
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .loss import compute_weighted_ce_loss

LOG = get_logger(__name__)


class AxolotlRCCATRTrainer(AxolotlTrainer):
    """
    RCCA-TR Trainer: Suppress-by-Default, Rescue-if-Useful.

    Memory footprint: only the active model (1× model size).
    Prior information comes from cached values in the batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LOG.info("Initializing RCCA-TR trainer...")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self._signature_columns:
            for col in ["prior_target_logp"]:
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
        Compute the RCCA-TR weighted CE loss.

        During training: uses suppress-by-default, rescue-if-useful weighting.
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
        inputs.pop("prior_margin", None)  # not used but may be in batch

        # Forward pass through active model
        outputs = model(**inputs)

        # During eval, use standard CE loss
        if not model.training:
            loss = outputs.loss if outputs.loss is not None else outputs[0]
            return (loss, outputs) if return_outputs else loss

        active_logits = outputs.logits  # (B, T, V)

        labels = inputs.get("labels", None)
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        B, T = labels.shape

        # Fallback: zeros if prior values not in batch
        if prior_target_logp is None:
            LOG.warning(
                "Prior cache values missing from batch. "
                "Falling back to zero tensors — RCCA-TR will degrade to standard CE."
            )
            prior_target_logp = torch.zeros(B, T, device=labels.device)

        # Compute unified weighted CE loss
        # All shifting is handled inside compute_weighted_ce_loss
        loss, intermediates = compute_weighted_ce_loss(
            active_logits=active_logits,
            labels=labels,
            prior_target_logp=prior_target_logp,
            tau_p=getattr(self.args, "rcca_tr_tau_p", 2.0) or 2.0,
            T_p=getattr(self.args, "rcca_tr_T_p", 1.0) or 1.0,
            tau_delta=getattr(self.args, "rcca_tr_tau_delta", 0.8) or 0.8,
            T_delta=getattr(self.args, "rcca_tr_T_delta", 1.0) or 1.0,
            w_min=getattr(self.args, "rcca_tr_w_min", 0.05) or 0.05,
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
