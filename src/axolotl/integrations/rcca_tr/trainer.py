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
Drift-Trust / token-role-triage Trainer.

Single trainer class for legacy modes and full-FT base-aware triage modes.

Mode selection is controlled by ``rcca_tr_mode`` in the training args.

For base-aware modes, frozen-base logits are obtained from a separate frozen
reference model. The full-FT module-aware method routes different losses to
attention, MLP, and other trainable parameter groups.

Paper-grade logging: emits w_t distribution stats, s_t, r_t, ce_t,
and suppressed/amplified token fractions on every logging step.
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.logging import get_logger

from .drift import DriftBuffer
from .loss import compute_module_routing_losses, compute_rcca_loss

LOG = get_logger(__name__)

# Canonical mode names (resolves legacy aliases at init time)
_LEGACY_MODE_MAP = {
    "drift_only": "drift_trust_s",
    "drift": "drift_trust_a",
}

# Modes that require the drift buffer
_DRIFT_MODES = {"drift_trust_s", "drift_trust_a"}

# Modes that require frozen-base logits
_BASE_AWARE_MODES = {
    "stm_top20",
    "soft_stm",
    "retention_kl",
    "learn_new",
    "module_aware_retention",
    "fullft_module_aware_retention",
    "attention_only_new",
}

_ROUTED_MODULE_MODES = {
    "module_aware_retention",
    "fullft_module_aware_retention",
}


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


def _arg(args, name: str, default):
    """Read TrainingArguments values while preserving explicit zeroes."""
    value = getattr(args, name, default)
    return default if value is None else value


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
        self.reference_model = None
        self._routed_param_cache = None
        self._logged_routed_param_groups = False

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

    def _unwrap_model(self, model):
        if hasattr(self, "accelerator") and self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        return model

    def _reference_model_name(self, model) -> str:
        configured = getattr(self.args, "rcca_tr_reference_model", None)
        if configured:
            return configured

        unwrapped = self._unwrap_model(model)
        config = getattr(unwrapped, "config", None)
        name = getattr(config, "_name_or_path", None)
        if name:
            return name

        raise RuntimeError(
            "Base-aware full-FT RCCA-TR modes require rcca_tr_reference_model "
            "or a model config with _name_or_path."
        )

    def _ensure_reference_model(self, model):
        """
        Lazily load the frozen pretrained reference used for token role gates.

        This intentionally uses a separate model for full fine-tuning. The old
        adapter-disabled shortcut is not used because the method now targets
        full-parameter updates, not LoRA deltas.
        """
        if self.reference_model is not None:
            return self.reference_model

        from transformers import AutoModelForCausalLM

        unwrapped = self._unwrap_model(model)
        ref_name = self._reference_model_name(unwrapped)
        param = next(unwrapped.parameters())
        dtype = param.dtype if param.is_floating_point() else torch.bfloat16
        device = param.device

        LOG.info(
            "Loading frozen RCCA-TR reference model for full FT: %s (dtype=%s, device=%s)",
            ref_name,
            dtype,
            device,
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            ref_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        reference_model.to(device)
        reference_model.eval()
        for ref_param in reference_model.parameters():
            ref_param.requires_grad_(False)

        self.reference_model = reference_model
        return self.reference_model

    def _get_base_logits(self, model, inputs) -> torch.Tensor:
        """Return frozen-base logits from the full-FT reference model."""
        reference_model = self._ensure_reference_model(model)
        base_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        with torch.no_grad():
            outputs = reference_model(**base_inputs)
        return outputs.logits.detach()

    def _forward_no_fp32_convert(self, model, inputs):
        """Forward pass while preserving bf16 logits for large-vocab losses."""
        import accelerate.utils.operations as accel_ops

        original_convert = accel_ops.convert_to_fp32
        accel_ops.convert_to_fp32 = lambda x: x
        try:
            return model(**inputs)
        finally:
            accel_ops.convert_to_fp32 = original_convert

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
        outputs = self._forward_no_fp32_convert(model, inputs)

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

        # ── Compute shared drift signal (for legacy drift modes) ──
        drift = None
        if self.mode in _DRIFT_MODES:
            with torch.no_grad():
                active_logp = self._get_logp(logits, labels, valid_mask)
            drift = self.drift_buffer.get_current_drift(active_logp, valid_mask)

        # ── Frozen-base logits for token role triage modes ──
        base_logits = None
        if self.mode in _BASE_AWARE_MODES:
            base_logits = self._get_base_logits(model, inputs)

        # ── Dispatch to unified loss ──
        loss, stats = compute_rcca_loss(
            mode=self.mode,
            logits=logits,
            labels=labels,
            drift=drift,
            base_logits=base_logits,
            self_tau=_arg(self.args, "rcca_tr_self_tau", 1.0),
            drift_tau=_arg(self.args, "rcca_tr_drift_tau", 1.0),
            w_min=_arg(self.args, "rcca_tr_w_min", 0.05),
            beta=_arg(self.args, "rcca_tr_beta", 0.5),
            gamma=_arg(self.args, "rcca_tr_drift_gamma", 1.0),
            reliability_tau=_arg(self.args, "rcca_tr_reliability_tau", 1.0),
            anchor_base=_arg(self.args, "rcca_tr_anchor_base", 0.1),
            anchor_lambda=_arg(self.args, "rcca_tr_anchor_lambda", 4.0),
            tau_old=_arg(self.args, "rcca_tr_tau_old", None),
            tau_new=_arg(self.args, "rcca_tr_tau_new", None),
            tau_noise=_arg(self.args, "rcca_tr_tau_noise", None),
            gate_temperature=_arg(self.args, "rcca_tr_gate_temperature", 1.0),
            old_quantile=_arg(self.args, "rcca_tr_old_quantile", 0.4),
            new_quantile=_arg(self.args, "rcca_tr_new_quantile", 0.6),
            noise_quantile=_arg(self.args, "rcca_tr_noise_quantile", 0.95),
            stm_keep_ratio=_arg(self.args, "rcca_tr_stm_keep_ratio", 0.8),
            lambda_acquire=_arg(self.args, "rcca_tr_lambda_acquire", 1.0),
            mu_noise=_arg(self.args, "rcca_tr_mu_noise", 1.0),
            rho_retention=_arg(self.args, "rcca_tr_rho_retention", 0.5),
            triage_w_floor=_arg(self.args, "rcca_tr_triage_w_floor", 0.1),
            triage_w_max=_arg(self.args, "rcca_tr_triage_w_max", 3.0),
            kl_beta=_arg(self.args, "rcca_tr_kl_beta", 0.05),
            kl_chunk_size=_arg(self.args, "rcca_tr_kl_chunk_size", 256),
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

    def _module_routing_kwargs(self):
        """Arguments for the full-FT attention/MLP/other routed objective."""
        return {
            "tau_old": _arg(self.args, "rcca_tr_tau_old", None),
            "tau_new": _arg(self.args, "rcca_tr_tau_new", None),
            "tau_noise": _arg(self.args, "rcca_tr_tau_noise", None),
            "gate_temperature": _arg(self.args, "rcca_tr_gate_temperature", 1.0),
            "old_quantile": _arg(self.args, "rcca_tr_old_quantile", 0.4),
            "new_quantile": _arg(self.args, "rcca_tr_new_quantile", 0.6),
            "noise_quantile": _arg(self.args, "rcca_tr_noise_quantile", 0.95),
            "attn_lambda_acquire": _arg(self.args, "rcca_tr_attn_lambda_acquire", 1.0),
            "attn_mu_noise": _arg(self.args, "rcca_tr_attn_mu_noise", 1.0),
            "attn_rho_retention": _arg(self.args, "rcca_tr_attn_rho_retention", 0.0),
            "attn_kl_beta": _arg(self.args, "rcca_tr_attn_kl_beta", 0.0),
            "mlp_lambda_acquire": _arg(self.args, "rcca_tr_mlp_lambda_acquire", 0.5),
            "mlp_mu_noise": _arg(self.args, "rcca_tr_mlp_mu_noise", 1.0),
            "mlp_rho_retention": _arg(self.args, "rcca_tr_mlp_rho_retention", 0.5),
            "mlp_kl_beta": _arg(self.args, "rcca_tr_mlp_kl_beta", 0.05),
            "other_lambda_acquire": _arg(self.args, "rcca_tr_other_lambda_acquire", 0.25),
            "other_mu_noise": _arg(self.args, "rcca_tr_other_mu_noise", 1.0),
            "other_rho_retention": _arg(self.args, "rcca_tr_other_rho_retention", 0.75),
            "other_kl_beta": _arg(self.args, "rcca_tr_other_kl_beta", 0.05),
            "w_floor": _arg(self.args, "rcca_tr_triage_w_floor", 0.1),
            "w_max": _arg(self.args, "rcca_tr_triage_w_max", 3.0),
            "kl_chunk_size": _arg(self.args, "rcca_tr_kl_chunk_size", 256),
        }

    def _compute_module_routing_step_loss(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        """Forward pass and loss bundle for full-FT module-routed training."""
        inputs = dict(inputs)
        if (
            self.args.sample_packing
            and "attention_mask" in inputs
            and "position_ids" in inputs
        ):
            del inputs["attention_mask"]

        labels = inputs.pop("labels", None)
        if labels is None:
            raise RuntimeError("full-FT module-routed RCCA-TR requires labels")

        if num_items_in_batch is None:
            num_items_in_batch = (labels != -100).sum().item()

        outputs = self._forward_no_fp32_convert(model, inputs)
        base_logits = self._get_base_logits(model, inputs)
        routed_losses, stats = compute_module_routing_losses(
            outputs.logits,
            labels,
            base_logits,
            **self._module_routing_kwargs(),
            num_items_in_batch=num_items_in_batch,
        )

        valid_mask = labels != -100
        if self.state.global_step % max(getattr(self.args, "logging_steps", 1), 1) == 0:
            self._log_stats(stats, valid_mask, labels)

        return routed_losses, stats, outputs, labels

    def _routed_param_groups(self, model):
        """Split trainable full-FT parameters into attention, MLP, and conservative other routes."""
        if self._routed_param_cache is not None:
            return self._routed_param_cache

        groups = {"attn": [], "mlp": [], "other": []}
        attn_patterns = (
            "self_attn",
            ".attn.",
            ".attention.",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            ".query",
            ".key",
            ".value",
        )
        mlp_patterns = (
            ".mlp",
            "feed_forward",
            "ffn",
            "gate_proj",
            "up_proj",
            "down_proj",
            "c_fc",
            "fc1",
            "fc2",
            "w1",
            "w2",
            "w3",
        )

        for name, param in self._unwrap_model(model).named_parameters():
            if not param.requires_grad:
                continue
            lower_name = name.lower()
            if any(pattern in lower_name for pattern in attn_patterns):
                groups["attn"].append(param)
            elif any(pattern in lower_name for pattern in mlp_patterns):
                groups["mlp"].append(param)
            else:
                groups["other"].append(param)

        if not self._logged_routed_param_groups:
            LOG.info(
                "RCCA-TR routed parameter groups: attention=%d, mlp=%d, other=%d",
                len(groups["attn"]),
                len(groups["mlp"]),
                len(groups["other"]),
            )
            self._logged_routed_param_groups = True

        self._routed_param_cache = groups
        return groups

    def _gradient_accumulation_scale(self) -> int:
        return max(
            int(
                getattr(
                    self,
                    "current_gradient_accumulation_steps",
                    getattr(self.args, "gradient_accumulation_steps", 1),
                )
                or 1
            ),
            1,
        )

    def _backward_routed_losses(self, model, routed_losses):
        """
        Backpropagate route-specific losses into only their matching modules.

        This keeps the method full-FT: all trainable parameters remain optimizer
        parameters, but each module receives the objective aligned with its role.
        """
        if self.use_apex:
            raise RuntimeError("full-FT module-routed RCCA-TR does not support Apex AMP")

        groups = self._routed_param_groups(model)
        active_routes = [
            ("attn", routed_losses["attn_loss"], groups["attn"]),
            ("mlp", routed_losses["mlp_loss"], groups["mlp"]),
            ("other", routed_losses["other_loss"], groups["other"]),
        ]
        active_routes = [
            (name, loss, params)
            for name, loss, params in active_routes
            if params
        ]
        if not active_routes:
            raise RuntimeError("full-FT module-routed RCCA-TR found no trainable parameters")

        all_routed_params = [
            param
            for _, _, params in active_routes
            for param in params
        ]
        original_requires_grad = [
            (param, param.requires_grad)
            for param in all_routed_params
        ]
        gas = self._gradient_accumulation_scale()

        try:
            for route_idx, (_, route_loss, route_params) in enumerate(active_routes):
                route_param_ids = {id(param) for param in route_params}
                for param in all_routed_params:
                    param.requires_grad_(id(param) in route_param_ids)

                loss = route_loss
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                loss = loss / gas
                self.accelerator.backward(
                    loss,
                    retain_graph=route_idx < len(active_routes) - 1,
                )
        finally:
            for param, requires_grad in original_requires_grad:
                param.requires_grad_(requires_grad)

    @override
    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        if self.mode not in _ROUTED_MODULE_MODES:
            return super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                routed_losses, _, outputs, _ = self._compute_module_routing_step_loss(
                    model,
                    inputs,
                    num_items_in_batch=num_items_in_batch,
                )

            self._backward_routed_losses(model, routed_losses)
            loss = routed_losses["loss"] / self._gradient_accumulation_scale()
            del inputs, outputs
            return loss.detach()

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
            "stm_top20": 4,
            "soft_stm": 5,
            "retention_kl": 6,
            "learn_new": 7,
            "module_aware_retention": 8,
            "attention_only_new": 9,
            "fullft_module_aware_retention": 10,
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
                high_thresh = (
                    1.5
                    if self.mode in {
                        "drift_trust_a",
                        "learn_new",
                        "module_aware_retention",
                        "fullft_module_aware_retention",
                        "attention_only_new",
                    }
                    else 0.8
                )
                log_dict["train/high_weight_frac"] = (w_masked > high_thresh).float().sum().item() / total

        for route_key, log_key in (
            ("w_attn_t", "attn_w"),
            ("w_mlp_t", "mlp_w"),
            ("w_other_t", "other_w"),
        ):
            if route_key in stats:
                route_stats = _summarize_tensor(stats[route_key], shift_mask)
                log_dict[f"train/{log_key}_mean"] = route_stats["mean"]

        if "s_t" in stats:
            s_stats = _summarize_tensor(stats["s_t"], shift_mask)
            log_dict["train/s_mean"] = s_stats["mean"]

        if "r_t" in stats:
            r_stats = _summarize_tensor(stats["r_t"], shift_mask)
            log_dict["train/r_mean"] = r_stats["mean"]

        if "R_t" in stats:
            r_stats = _summarize_tensor(stats["R_t"], shift_mask)
            log_dict["train/retention_gate_mean"] = r_stats["mean"]

        if "A_t" in stats:
            a_stats = _summarize_tensor(stats["A_t"], shift_mask)
            log_dict["train/acquisition_gate_mean"] = a_stats["mean"]

        if "N_t" in stats:
            n_stats = _summarize_tensor(stats["N_t"], shift_mask)
            log_dict["train/noise_gate_mean"] = n_stats["mean"]

        if "ce_t" in stats:
            ce_stats = _summarize_tensor(stats["ce_t"], shift_mask)
            log_dict["train/ce_mean"] = ce_stats["mean"]

        if "base_nll_t" in stats:
            base_stats = _summarize_tensor(stats["base_nll_t"], shift_mask)
            log_dict["train/base_nll_mean"] = base_stats["mean"]

        if "kl_t" in stats:
            kl_stats = _summarize_tensor(stats["kl_t"], shift_mask)
            log_dict["train/base_kl_mean"] = kl_stats["mean"]

        for key in (
            "tau_old",
            "tau_new",
            "tau_noise",
            "ce_loss",
            "kl_loss",
            "attn_loss",
            "attn_ce_loss",
            "attn_kl_loss",
            "mlp_loss",
            "mlp_ce_loss",
            "mlp_kl_loss",
            "other_loss",
            "other_ce_loss",
            "other_kl_loss",
        ):
            value = stats.get(key)
            if torch.is_tensor(value) and value.numel() == 1:
                log_dict[f"train/{key}"] = value.item()

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
