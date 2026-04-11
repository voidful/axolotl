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
Drift Trainer.

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


class AxolotlDriftTrainer(AxolotlTrainer):
    """
    Drift Trust-Region Trainer.

    Memory footprint: only the active model (1× model size).
    Evidence drift is tracked via a lightweight DriftBuffer using
    the active model's CE as the sole drift signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        LOG.info("Initializing Drift trainer...")

        # Initialize drift buffer
        ema_decay = getattr(self.args, "drift_ema_decay", 0.999) or 0.999
        drift_gamma = getattr(self.args, "drift_gamma", 1.0) or 1.0
        self.drift_buffer = DriftBuffer(decay=ema_decay, gamma=drift_gamma)

        LOG.info(
            "Drift trainer initialized (drift_decay=%.4f, drift_gamma=%.2f)",
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
        Compute the Drift trust-region loss.

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

        # Pop labels so the model forward skips internal loss (avoids fp32 upcast)
        labels = inputs.pop("labels", None)

        if num_items_in_batch is None and labels is not None:
            num_items_in_batch = (labels != -100).sum().item()

        # Forward pass — bypass accelerate's fp32 upcast to keep logits in bf16
        import accelerate.utils.operations as accel_ops
        original_convert = accel_ops.convert_to_fp32
        accel_ops.convert_to_fp32 = lambda x: x
        try:
            outputs = model(**inputs)
        finally:
            accel_ops.convert_to_fp32 = original_convert

        # During eval, use standard CE loss (trust-region doesn't apply)
        if not model.training:
            if labels is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = outputs.loss if outputs.loss is not None else outputs[0]
            return (loss, outputs) if return_outputs else loss

        active_logits = outputs.logits  # (B, T, V) — stays in bf16

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        B, T = labels.shape
        valid_mask = labels != -100

        # 1. Compute drift-based reliability
        # Avoid full-vocab log_softmax: use logsumexp + gather (only (B,T) intermediates)
        with torch.no_grad():
            safe_labels = labels.clamp(min=0)
            target_logit = active_logits.gather(
                dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)
            lse = torch.logsumexp(active_logits, dim=-1)  # (B, T)
            active_target_logp = (target_logit - lse) * valid_mask.float()

        drift = self.drift_buffer.get_current_drift(
            active_target_logp=active_target_logp,
            valid_mask=valid_mask,
            per_sample=getattr(self.args, "drift_per_sample", False) or False,
        )

        r_t = compute_reliability_from_drift(
            drift=drift,
            gamma=getattr(self.args, "drift_gamma", 1.0) or 1.0,
            tau=getattr(self.args, "drift_reliability_tau", 1.0) or 1.0,
        )

        # 2. Compute trust-region loss
        loss, _ = compute_trust_region_loss(
            active_logits=active_logits,
            labels=labels,
            r_t=r_t,
            kl_lambda=getattr(self.args, "drift_kl_lambda", 1.0) or 1.0,
            anchor_weight=getattr(self.args, "drift_anchor_weight", 0.5) or 0.0,
            epsilon_min=getattr(self.args, "drift_epsilon_min", 0.01) or 0.01,
            epsilon_max=getattr(self.args, "drift_epsilon_max", 1.0) or 1.0,
            use_smooth=getattr(self.args, "drift_use_smooth_objective", True),
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
        """
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
