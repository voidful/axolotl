"""
Unit tests for RCCA-TR loss functions and utilities.
"""

import torch
import pytest
from axolotl.integrations.rcca_tr.loss import (
    compute_conflict_score,
    compute_stability,
    compute_evidence_drift,
    compute_reliability,
    compute_trust_region_loss,
)
from axolotl.integrations.rcca_tr.ema import update_ema_model


class TestConflictScore:
    """Tests for compute_conflict_score."""

    def test_shape(self):
        """Output shape matches (B, T)."""
        B, T, V = 2, 10, 100
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[:, 0] = -100  # some ignored tokens

        alpha = compute_conflict_score(frozen_logits, labels)
        assert alpha.shape == (B, T)

    def test_range(self):
        """Output values are in [0, 1]."""
        B, T, V = 4, 20, 50
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        alpha = compute_conflict_score(frozen_logits, labels)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_ignored_tokens_zero(self):
        """Ignored tokens (label=-100) should have α=0."""
        B, T, V = 2, 10, 50
        frozen_logits = torch.randn(B, T, V)
        labels = torch.full((B, T), -100, dtype=torch.long)

        alpha = compute_conflict_score(frozen_logits, labels)
        assert torch.all(alpha == 0.0)

    def test_high_conflict_for_wrong_prediction(self):
        """When frozen model strongly predicts wrong token, conflict should be high."""
        B, T, V = 1, 5, 10
        # Make frozen model confident on token 0
        frozen_logits = torch.zeros(B, T, V)
        frozen_logits[:, :, 0] = 10.0  # strongly predicts token 0
        # Ground truth is token 5 (different from prediction)
        labels = torch.full((B, T), 5, dtype=torch.long)

        alpha = compute_conflict_score(frozen_logits, labels, lambda1=1.0, lambda2=0.5)
        # Conflict should be relatively high (above 0.5)
        assert alpha.mean() > 0.5


class TestStability:
    """Tests for compute_stability."""

    def test_identical_perturbations_high_stability(self):
        """When all perturbations are identical, stability should be ~1."""
        B, T, V = 2, 10, 50
        ref = torch.randn(B, T, V)
        perturbations = [ref.clone() for _ in range(3)]

        stability = compute_stability(perturbations, ref)
        assert stability.shape == (B, T)
        assert torch.allclose(stability, torch.ones_like(stability), atol=1e-5)

    def test_random_perturbations_lower_stability(self):
        """When perturbations are very different, stability should be lower."""
        B, T, V = 2, 10, 50
        ref = torch.randn(B, T, V)
        perturbations = [torch.randn(B, T, V) * 10 for _ in range(5)]

        stability = compute_stability(perturbations, ref)
        assert stability.mean() < 0.9  # should be lower than perfect


class TestEvidenceDrift:
    """Tests for compute_evidence_drift."""

    def test_no_drift(self):
        """When EMA = frozen, evidence reliability should be ~1."""
        B, T, V = 2, 10, 50
        logits = torch.randn(B, T, V)

        r_evi = compute_evidence_drift(logits, logits)
        assert r_evi.shape == (B, T)
        assert torch.allclose(r_evi, torch.ones_like(r_evi), atol=1e-5)

    def test_large_drift_low_reliability(self):
        """When EMA and frozen are very different, reliability should be low."""
        B, T, V = 2, 10, 50
        ema_logits = torch.randn(B, T, V) * 10
        frozen_logits = torch.randn(B, T, V) * 10

        r_evi = compute_evidence_drift(ema_logits, frozen_logits)
        assert r_evi.mean() < 0.9


class TestReliability:
    """Tests for compute_reliability."""

    def test_range(self):
        """Output should be in [0, 1]."""
        B, T = 4, 20
        stability = torch.rand(B, T)
        evidence = torch.rand(B, T)

        r_t = compute_reliability(stability, evidence)
        assert r_t.shape == (B, T)
        assert r_t.min() >= 0.0
        assert r_t.max() <= 1.0


class TestTrustRegionLoss:
    """Tests for compute_trust_region_loss."""

    def test_produces_scalar(self):
        """Loss should be a scalar."""
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)

        loss = compute_trust_region_loss(
            active_logits, frozen_logits, labels, alpha_t, r_t
        )
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows(self):
        """Gradient should flow through the loss."""
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)

        loss = compute_trust_region_loss(
            active_logits, frozen_logits, labels, alpha_t, r_t
        )
        loss.backward()
        assert active_logits.grad is not None
        assert not torch.all(active_logits.grad == 0)

    def test_smooth_vs_hinge(self):
        """Both smooth and hinge variants should produce valid losses."""
        B, T, V = 2, 10, 50
        active_logits = torch.randn(B, T, V, requires_grad=True)
        frozen_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        alpha_t = torch.rand(B, T)
        r_t = torch.rand(B, T)

        loss_smooth = compute_trust_region_loss(
            active_logits, frozen_logits, labels, alpha_t, r_t, use_smooth=True
        )
        loss_hinge = compute_trust_region_loss(
            active_logits.detach().requires_grad_(True), frozen_logits, labels,
            alpha_t, r_t, use_smooth=False
        )

        assert not torch.isnan(loss_smooth)
        assert not torch.isnan(loss_hinge)


class TestEMAUpdate:
    """Tests for EMA parameter update."""

    def test_ema_moves_toward_active(self):
        """After EMA update, EMA params should be closer to active params."""
        import torch.nn as nn

        model = nn.Linear(10, 10)
        ema_model = nn.Linear(10, 10)

        # Initialize differently
        with torch.no_grad():
            model.weight.fill_(1.0)
            ema_model.weight.fill_(0.0)

        initial_dist = (model.weight - ema_model.weight).norm().item()

        update_ema_model(ema_model, model, decay=0.9)

        final_dist = (model.weight - ema_model.weight).norm().item()

        assert final_dist < initial_dist

    def test_ema_value_correctness(self):
        """EMA update formula: ema = decay * ema + (1-decay) * active."""
        import torch.nn as nn

        model = nn.Linear(10, 10, bias=False)
        ema_model = nn.Linear(10, 10, bias=False)

        with torch.no_grad():
            model.weight.fill_(1.0)
            ema_model.weight.fill_(0.0)

        decay = 0.9
        update_ema_model(ema_model, model, decay=decay)

        expected = 0.0 * decay + 1.0 * (1 - decay)  # = 0.1
        assert torch.allclose(
            ema_model.weight, torch.full_like(ema_model.weight, expected), atol=1e-6
        )
