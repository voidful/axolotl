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
Drift-Trust Trainer.

Single trainer class for all four modes:
  {ce, hardness, drift_trust_s, drift_trust_a}

Mode selection is controlled by ``rcca_tr_mode`` in the training args.

Only one model lives in GPU memory — the active model being trained.
Evidence drift is tracked via a lightweight DriftBuffer using the
active model's own log-probabilities as the drift signal.

Paper-grade logging: emits w_t distribution stats, s_t, r_t, ce_t,
and suppressed/amplified token fractions on every logging step.
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .drift import DriftBuffer
from .loss import compute_rcca_loss

LOG = get_logger(__name__)

# Canonical mode names (resolves legacy aliases at init time)
_LEGACY_MODE_MAP = {
    "drift_only": "drift_trust_s",
    "drift": "drift_trust_a",
}

# Modes that require the drift buffer
_DRIFT_MODES = {"drift_trust_s", "drift_trust_a"}


def _summarize_tensor(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute mean, q10, q50, q90 of masked tensor for logging."""
    vals = x[mask]
    if vals.numel() == 0:
        return {"mean": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0}
    return {
        "mean": vals.mean().item(),
        "q10": torch.quantile(vals.float(), 0.1).item(),
        "q50": torch.quantile(vals.float(), 0.5).item(),
        "q90": torch.quantile(vals.float(), 0.9).item(),
    }


class AxolotlRCCATRTrainer(AxolotlTrainer):
    """
    Drift-Trust Trainer — unified trainer for all four modes.

    Memory footprint: only the active model (1× model size).
    Evidence drift is tracked via a lightweight DriftBuffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        raw_mode = getattr(self.args, "rcca_tr_mode", "drift_trust_s") or "drift_trust_s"
        self.mode = _LEGACY_MODE_MAP.get(raw_mode, raw_mode)
        ema_decay = getattr(self.args, "rcca_tr_ema_decay", 0.999) or 0.999

        # Initialize drift buffer (used by drift_trust_s and drift_trust_a)
        self.drift_buffer = DriftBuffer(decay=ema_decay)

        LOG.info(
            "Drift-Trust trainer initialized (mode=%s, ema_decay=%.4f)",
            self.mode,
            ema_decay,
        )

    def _get_logp(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficiently compute log p_θ(y_t) without full-vocab log_softmax.

        Uses logsumexp + gather to produce (B, T) log-probs in O(V) instead
        of materializing an (B, T, V) tensor.
        """
        safe_labels = labels.clamp(min=0)
        target_logit = logits.gather(
            dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)
        lse = torch.logsumexp(logits, dim=-1)  # (B, T)
        return (target_logit - lse) * valid_mask.float()

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute Drift-Trust loss using the unified dispatch.

        During training: uses the configured mode (ce/hardness/drift_trust_s/drift_trust_a).
        During eval: falls back to standard CE for proper eval_loss metric.
        """
        # Handle sample packing
        if (
            self.args.sample_packing
            and "attention_mask" in inputs
            and "position_ids" in inputs
        ):
            del inputs["attention_mask"]

        # Pop labels so model forward skips internal loss (avoids fp32 upcast)
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

        # During eval: standard CE
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

        logits = outputs.logits  # (B, T, V)

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        valid_mask = labels != -100

        # ── Compute shared drift signal (for drift modes) ──
        drift = None
        if self.mode in _DRIFT_MODES:
            with torch.no_grad():
                active_logp = self._get_logp(logits, labels, valid_mask)
            drift = self.drift_buffer.get_current_drift(active_logp, valid_mask)

        # ── Dispatch to unified loss ──
        loss, stats = compute_rcca_loss(
            mode=self.mode,
            logits=logits,
            labels=labels,
            drift=drift,
            self_tau=getattr(self.args, "rcca_tr_self_tau", 1.0) or 1.0,
            drift_tau=getattr(self.args, "rcca_tr_drift_tau", 1.0) or 1.0,
            w_min=getattr(self.args, "rcca_tr_w_min", 0.05) or 0.05,
            beta=getattr(self.args, "rcca_tr_beta", 0.5) or 0.5,
            gamma=getattr(self.args, "rcca_tr_drift_gamma", 1.0) or 1.0,
            reliability_tau=getattr(self.args, "rcca_tr_reliability_tau", 1.0) or 1.0,
            anchor_base=getattr(self.args, "rcca_tr_anchor_base", 0.1) or 0.1,
            anchor_lambda=getattr(self.args, "rcca_tr_anchor_lambda", 4.0) or 4.0,
            num_items_in_batch=num_items_in_batch,
        )

        # ── Update drift buffer ──
        if self.mode in _DRIFT_MODES:
            with torch.no_grad():
                active_logp = self._get_logp(logits, labels, valid_mask)
            self.drift_buffer.step(active_logp, valid_mask)

        # ── Paper-grade logging ──
        if self.state.global_step % max(getattr(self.args, "logging_steps", 1), 1) == 0:
            self._log_stats(stats, valid_mask, labels)

        return (loss, outputs) if return_outputs else loss

    def _log_stats(
        self,
        stats: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Log paper-grade training statistics.

        Emits:
          - train/mode (0=ce, 1=hardness, 2=drift_trust_s, 3=drift_trust_a)
          - train/w_mean, w_q10, w_q50, w_q90
          - train/s_mean, r_mean (when available)
          - train/ce_mean
          - train/suppressed_frac (w_t < 0.2)
          - train/high_weight_frac (w_t > 0.8 for S, w_t > 1.5 for A)
          - train/drift_buffer_mean
        """
        shift_mask = labels[..., 1:] != -100

        mode_id = {
            "ce": 0,
            "hardness": 1,
            "drift_trust_s": 2,
            "drift_trust_a": 3,
        }.get(self.mode, -1)

        log_dict = {"train/mode": mode_id}

        if "w_t" in stats:
            w_stats = _summarize_tensor(stats["w_t"], shift_mask)
            log_dict.update({
                "train/w_mean": w_stats["mean"],
                "train/w_q10": w_stats["q10"],
                "train/w_q50": w_stats["q50"],
                "train/w_q90": w_stats["q90"],
            })

            # Fraction of tokens suppressed / amplified
            w_masked = stats["w_t"][shift_mask]
            if w_masked.numel() > 0:
                total = w_masked.numel()
                log_dict["train/suppressed_frac"] = (w_masked < 0.2).float().sum().item() / total
                # Use regime-appropriate threshold for "high weight"
                high_thresh = 1.5 if self.mode == "drift_trust_a" else 0.8
                log_dict["train/high_weight_frac"] = (w_masked > high_thresh).float().sum().item() / total

        if "s_t" in stats:
            s_stats = _summarize_tensor(stats["s_t"], shift_mask)
            log_dict["train/s_mean"] = s_stats["mean"]

        if "r_t" in stats:
            r_stats = _summarize_tensor(stats["r_t"], shift_mask)
            log_dict["train/r_mean"] = r_stats["mean"]

        if "ce_t" in stats:
            ce_stats = _summarize_tensor(stats["ce_t"], shift_mask)
            log_dict["train/ce_mean"] = ce_stats["mean"]

        # Drift buffer state
        log_dict["train/drift_buffer_mean"] = self.drift_buffer.state

        self.log(log_dict)

    @override
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Override prediction_step to use standard CE loss during eval."""
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
