"""
Unit tests for Drift-Focal loss, EMA reference buffer, and trainer wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from axolotl.integrations.drift import DriftPlugin
from axolotl.integrations.drift.drift import DriftMeanBuffer
from axolotl.integrations.drift.loss import compute_drift_focal_loss
from axolotl.integrations.drift.trainer import AxolotlDriftTrainer


def _make_batch(
    batch_size: int = 2,
    seq_len: int = 6,
    vocab_size: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, 0] = -100
    labels[:, -1] = -100
    return logits, labels


def _expected_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class TestDriftFocalLoss:
    """Behavioral tests for the one-parameter Drift-Focal loss."""

    def test_gamma_zero_matches_cross_entropy(self):
        logits, labels = _make_batch()

        loss, stats = compute_drift_focal_loss(
            active_logits=logits,
            labels=labels,
            reference_target_logp=-2.5,
            gamma=0.0,
        )

        expected = _expected_ce(logits, labels)
        mask = stats["shift_mask"]
        assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            stats["w_t"][mask],
            torch.ones_like(stats["w_t"][mask]),
            atol=1e-6,
        )

    def test_scalar_reference_keeps_absolute_drift(self):
        logits, labels = _make_batch()

        _, stats = compute_drift_focal_loss(
            active_logits=logits,
            labels=labels,
            reference_target_logp=-1.75,
            gamma=1.0,
        )

        mask = stats["shift_mask"]
        delta_t = stats["delta_t"]
        std_delta = delta_t[mask].std(unbiased=False)
        expected = delta_t / (std_delta + 1e-6)
        centered = (delta_t - delta_t[mask].mean()) / (std_delta + 1e-6)

        assert stats["reference_type"] == "scalar"
        assert torch.allclose(
            stats["delta_norm"][mask],
            expected[mask],
            atol=1e-6,
            rtol=1e-6,
        )
        assert not torch.allclose(
            stats["delta_norm"][mask],
            centered[mask],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_token_reference_uses_full_zscore(self):
        logits, labels = _make_batch()
        reference = torch.zeros(labels.shape, dtype=logits.dtype)

        _, stats = compute_drift_focal_loss(
            active_logits=logits,
            labels=labels,
            reference_target_logp=reference,
            gamma=1.0,
        )

        mask = stats["shift_mask"]
        assert stats["reference_type"] == "token"
        assert abs(stats["delta_norm"][mask].mean().item()) < 1e-6

    def test_mean_preserving_weights_average_to_one(self):
        logits, labels = _make_batch()

        _, stats = compute_drift_focal_loss(
            active_logits=logits,
            labels=labels,
            reference_target_logp=-2.0,
            gamma=2.0,
        )

        mask = stats["shift_mask"]
        assert torch.allclose(
            stats["w_t"][mask].mean(),
            torch.ones((), dtype=stats["w_t"].dtype),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_attached_weights_change_gradient_path(self):
        logits_a, labels = _make_batch()
        logits_b = logits_a.detach().clone().requires_grad_(True)

        loss_detached, _ = compute_drift_focal_loss(
            active_logits=logits_a,
            labels=labels,
            reference_target_logp=-2.0,
            gamma=2.0,
            detach_weights=True,
        )
        loss_attached, _ = compute_drift_focal_loss(
            active_logits=logits_b,
            labels=labels,
            reference_target_logp=-2.0,
            gamma=2.0,
            detach_weights=False,
        )

        grad_detached = torch.autograd.grad(loss_detached, logits_a)[0]
        grad_attached = torch.autograd.grad(loss_attached, logits_b)[0]

        assert torch.isfinite(grad_detached).all()
        assert torch.isfinite(grad_attached).all()
        assert not torch.allclose(grad_detached, grad_attached)


class TestDriftMeanBuffer:
    """Tests for the scalar EMA reference buffer."""

    def test_initial_step_uses_first_batch_mean(self):
        buf = DriftMeanBuffer(decay=0.9)
        active = torch.tensor([[-2.0, -4.0, 0.0]])
        mask = torch.tensor([[True, True, False]])

        buf.step(active, mask)

        assert buf.initialized is True
        assert buf.get_reference() == pytest.approx(-3.0)

    def test_subsequent_steps_apply_ema(self):
        buf = DriftMeanBuffer(decay=0.5)
        mask = torch.ones(1, 2, dtype=torch.bool)

        buf.step(torch.tensor([[-4.0, -2.0]]), mask)
        buf.step(torch.tensor([[-2.0, 0.0]]), mask)

        assert buf.get_reference() == pytest.approx(-2.0)


class DummyOutput:
    """Minimal transformers-style output object for trainer tests."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.loss = None

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.logits
        raise IndexError(idx)


class DummyModel(nn.Module):
    """Tiny model that only accepts standard LM inputs."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits_param = nn.Parameter(logits.detach().clone())

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        return DummyOutput(self.logits_param)


class TestDriftTrainer:
    """Trainer-level tests for EMA/prior reference wiring."""

    def _make_trainer(self, reference_mode: str) -> AxolotlDriftTrainer:
        trainer = object.__new__(AxolotlDriftTrainer)
        trainer.args = SimpleNamespace(
            sample_packing=False,
            drift_reference_mode=reference_mode,
            drift_reference_key="reference_target_logp",
            drift_ema_decay=0.9,
            drift_gamma=1.0,
            drift_detach_weights=True,
            drift_eps=1e-6,
        )
        trainer.drift_buffer = DriftMeanBuffer(decay=0.9)
        return trainer

    def test_ema_mode_updates_running_reference(self):
        logits, labels = _make_batch()
        trainer = self._make_trainer("ema")
        model = DummyModel(logits)

        loss = AxolotlDriftTrainer.compute_loss(
            trainer,
            model,
            {"input_ids": torch.zeros_like(labels), "labels": labels.clone()},
        )

        assert torch.isfinite(loss)
        assert trainer.drift_buffer.initialized is True

    def test_prior_mode_consumes_reference_tensor_before_model_forward(self):
        logits, labels = _make_batch()
        trainer = self._make_trainer("prior")
        model = DummyModel(logits)

        loss = AxolotlDriftTrainer.compute_loss(
            trainer,
            model,
            {
                "input_ids": torch.zeros_like(labels),
                "labels": labels.clone(),
                "reference_target_logp": torch.zeros(
                    labels.shape,
                    dtype=logits.dtype,
                ),
            },
        )

        assert torch.isfinite(loss)
        assert trainer.drift_buffer.initialized is False


def test_plugin_imports():
    plugin = DriftPlugin()
    assert plugin.get_input_args() == "axolotl.integrations.drift.DriftArgs"
    assert (
        plugin.get_training_args_mixin()
        == "axolotl.integrations.drift.args.DriftTrainingArgsMixin"
    )
