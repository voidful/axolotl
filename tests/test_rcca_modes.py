"""
Regression tests for RCCA-TR integration.

Tests:
  1. All four modes produce a valid scalar loss.
  2. drift_only weights are bounded in [w_min, 1.0].
  3. DriftBuffer running mean updates monotonically with consistent input.
  4. Unified dispatch raises on unknown mode.
"""

import pytest
import torch

from axolotl.integrations.rcca_tr.drift import DriftBuffer
from axolotl.integrations.rcca_tr.loss import (
    compute_ce_loss,
    compute_drift_only_loss,
    compute_drift_legacy_loss,
    compute_hardness_loss,
    compute_rcca_loss,
    compute_self_paced_score,
    compute_drift_score,
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    """Create minimal dummy data for loss computation."""
    torch.manual_seed(42)
    B, T, V = 2, 16, 100
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    # Mask some positions as padding
    labels[:, -3:] = -100
    return logits, labels


@pytest.fixture
def dummy_drift(dummy_data):
    """Create dummy drift tensor matching the data shape."""
    logits, labels = dummy_data
    B, T = labels.shape
    return torch.randn(B, T) * 0.5


# ── Test 1: All modes produce valid scalar loss ───────────────────────

class TestAllModesForward:
    """Every mode must produce a finite scalar loss."""

    def test_ce_mode(self, dummy_data):
        logits, labels = dummy_data
        loss, stats = compute_ce_loss(logits, labels)
        assert loss.dim() == 0, "Loss must be scalar"
        assert torch.isfinite(loss), "Loss must be finite"
        assert "ce_t" in stats

    def test_hardness_mode(self, dummy_data):
        logits, labels = dummy_data
        loss, stats = compute_hardness_loss(logits, labels, self_tau=1.0, w_min=0.05)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "s_t" in stats and "w_t" in stats

    def test_drift_only_mode(self, dummy_data, dummy_drift):
        logits, labels = dummy_data
        loss, stats = compute_drift_only_loss(
            logits, labels, dummy_drift,
            self_tau=1.0, drift_tau=1.0, w_min=0.05, beta=0.5,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "s_t" in stats and "r_t" in stats and "w_t" in stats

    def test_drift_legacy_mode(self, dummy_data, dummy_drift):
        logits, labels = dummy_data
        loss, stats = compute_drift_legacy_loss(
            logits, labels, dummy_drift,
            gamma=1.0, reliability_tau=1.0, kl_lambda=4.0, anchor_weight=0.1,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "r_t" in stats and "w_t" in stats

    def test_unified_dispatch_all_modes(self, dummy_data, dummy_drift):
        logits, labels = dummy_data
        for mode in ("ce", "hardness", "drift_only", "drift"):
            loss, stats = compute_rcca_loss(
                mode=mode, logits=logits, labels=labels, drift=dummy_drift,
            )
            assert loss.dim() == 0, f"{mode} did not return scalar"
            assert torch.isfinite(loss), f"{mode} returned non-finite loss"


# ── Test 2: Weight bounds ─────────────────────────────────────────────

class TestWeightBounds:
    """drift_only weights must be in [w_min, 1.0]."""

    @pytest.mark.parametrize("w_min", [0.0, 0.05, 0.1, 0.5])
    def test_w_range(self, dummy_data, dummy_drift, w_min):
        logits, labels = dummy_data
        _, stats = compute_drift_only_loss(
            logits, labels, dummy_drift, w_min=w_min,
        )
        w_t = stats["w_t"]
        mask = labels[..., 1:] != -100

        # Only check valid positions
        w_valid = w_t[mask]
        assert w_valid.min() >= w_min - 1e-6, f"w_t below w_min: {w_valid.min()}"
        assert w_valid.max() <= 1.0 + 1e-6, f"w_t above 1.0: {w_valid.max()}"


# ── Test 3: DriftBuffer update monotonicity ───────────────────────────

class TestDriftBuffer:
    """DriftBuffer running mean moves toward the batch mean."""

    def test_running_mean_converges_to_batch_mean(self):
        """Running mean should converge toward the constant batch mean."""
        buf = DriftBuffer(decay=0.9)
        mask = torch.ones(1, 10, dtype=torch.bool)

        # Feed constant value repeatedly, running mean should converge
        for _ in range(50):
            buf.step(torch.full((1, 10), -3.0), mask)

        # Should be close to -3.0 after many steps
        assert abs(buf.state - (-3.0)) < 0.1, (
            f"Running mean did not converge to -3.0: {buf.state}"
        )

    def test_running_mean_tracks_direction_change(self):
        """Running mean should move toward new batch mean when input changes."""
        buf = DriftBuffer(decay=0.9)
        mask = torch.ones(1, 10, dtype=torch.bool)

        # Converge to -5.0
        for _ in range(50):
            buf.step(torch.full((1, 10), -5.0), mask)
        old_mean = buf.state

        # Switch to -1.0
        buf.step(torch.full((1, 10), -1.0), mask)
        new_mean = buf.state

        # Should have moved toward -1.0 (i.e., increased)
        assert new_mean > old_mean, (
            f"Running mean did not move toward new batch mean: {old_mean} -> {new_mean}"
        )

    def test_drift_sign(self):
        buf = DriftBuffer(decay=0.9)
        mask = torch.ones(1, 5, dtype=torch.bool)

        # Converge running mean to -3.0
        for _ in range(100):
            buf.step(torch.full((1, 5), -3.0), mask)

        # Easy token (higher log-prob than mean) → positive drift
        easy = torch.full((1, 5), -1.0)
        drift_easy = buf.get_current_drift(easy, mask)
        assert (drift_easy > 0).all(), "Easy tokens should have positive drift"

        # Hard token (lower log-prob than mean) → negative drift
        hard = torch.full((1, 5), -6.0)
        drift_hard = buf.get_current_drift(hard, mask)
        assert (drift_hard < 0).all(), "Hard tokens should have negative drift"

    def test_masked_positions_zeroed(self):
        buf = DriftBuffer(decay=0.9)
        mask = torch.tensor([[True, True, False, False, True]])
        logp = torch.randn(1, 5)

        drift = buf.get_current_drift(logp, mask)
        assert (drift[~mask] == 0).all(), "Masked positions must have zero drift"


# ── Test 4: Score functions ───────────────────────────────────────────

class TestScoreFunctions:
    def test_self_paced_range(self):
        logp = torch.randn(2, 10)
        mask = torch.ones(2, 10, dtype=torch.bool)
        s_t = compute_self_paced_score(logp, mask, tau=1.0)
        assert s_t.min() >= 0.0
        assert s_t.max() <= 1.0

    def test_drift_score_range(self):
        drift = torch.randn(2, 10)
        mask = torch.ones(2, 10, dtype=torch.bool)
        r_t = compute_drift_score(drift, mask, tau=1.0)
        assert r_t.min() >= 0.0
        assert r_t.max() <= 1.0


# ── Test 5: Error handling ────────────────────────────────────────────

class TestErrorHandling:
    def test_unknown_mode_raises(self, dummy_data):
        logits, labels = dummy_data
        with pytest.raises(ValueError, match="Unknown rcca_tr_mode"):
            compute_rcca_loss(mode="banana", logits=logits, labels=labels)

    def test_drift_only_without_drift_raises(self, dummy_data):
        logits, labels = dummy_data
        with pytest.raises(ValueError, match="requires drift tensor"):
            compute_rcca_loss(mode="drift_only", logits=logits, labels=labels, drift=None)
