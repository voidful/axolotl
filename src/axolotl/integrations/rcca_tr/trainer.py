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
RCCA-TR Trainer (no-cache variant).

Only one model lives in GPU memory — the active model being trained.
Evidence drift is tracked via a lightweight drift buffer using the
active model's own CE as the drift signal.

On each step, it computes:
  - Drift-based reliability r_t
  - Trust-region loss: λ · r_t · CE
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .drift import DriftBuffer
from .loss import (
    compute_reliability_from_drift,
    compute_trust_region_loss,
)

LOG = get_logger(__name__)


class AxolotlRCCATRTrainer(AxolotlTrainer):
    """
    RCCA-TR Trust-Region Trainer (no-cache variant).

    Memory footprint: only the active model (1× model size).
    Evidence drift is tracked via a lightweight DriftBuffer using
    the active model's CE as the sole drift signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        LOG.info("Initializing RCCA-TR trainer...")

        # Initialize drift buffer
        ema_decay = getattr(self.args, "rcca_tr_ema_decay", 0.999) or 0.999
        drift_gamma = getattr(self.args, "rcca_tr_drift_gamma", 1.0) or 1.0
        self.drift_buffer = DriftBuffer(decay=ema_decay, gamma=drift_gamma)

        LOG.info(
            "RCCA-TR trainer initialized (drift_decay=%.4f, drift_gamma=%.2f)",
            ema_decay,
            drift_gamma,
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
        Compute the RCCA-TR trust-region loss.

        During training: uses drift buffer + KL proxy.
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

        # 1. Compute drift-based reliability
        # Get active model's log p(y_t) for drift estimation
        with torch.no_grad():
            active_log_probs = F.log_softmax(active_logits, dim=-1)
            safe_labels = labels.clamp(min=0)
            active_target_logp = active_log_probs.gather(
                dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)
            active_target_logp = active_target_logp * valid_mask.float()

        drift = self.drift_buffer.get_current_drift(
            active_target_logp=active_target_logp,
            valid_mask=valid_mask,
        )

        r_t = compute_reliability_from_drift(
            drift=drift,
            gamma=getattr(self.args, "rcca_tr_drift_gamma", 1.0) or 1.0,
            tau=getattr(self.args, "rcca_tr_reliability_tau", 1.0) or 1.0,
        )

        # 2. Compute trust-region loss
        loss, _ = compute_trust_region_loss(
            active_logits=active_logits,
            labels=labels,
            r_t=r_t,
            kl_lambda=getattr(self.args, "rcca_tr_kl_lambda", 1.0) or 1.0,
            epsilon_min=getattr(self.args, "rcca_tr_epsilon_min", 0.01) or 0.01,
            epsilon_max=getattr(self.args, "rcca_tr_epsilon_max", 1.0) or 1.0,
            use_smooth=getattr(self.args, "rcca_tr_use_smooth_objective", True),
            num_items_in_batch=num_items_in_batch,
        )

        # 3. Update drift buffer with this step's values
        self.drift_buffer.step(
            active_target_logp=active_target_logp,
            valid_mask=valid_mask,
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
