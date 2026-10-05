"""Metrics on hand-built cases."""

import numpy as np
import pytest

from benchmarks.metrics import score

TRUTH = np.linspace(1.0, 2.0, 100)


def test_success_claimed_and_covered():
    s = score(TRUTH + 0.001, np.full(100, 0.005), TRUTH, rel_tol=0.05)
    # sigma/eps_hat = 0.005 / (0.05 * |m_y|) <= 0.1 at most; all errors 0.2 sigma
    assert s["claimed"] and s["success"] and not s["false_claim"]
    assert s["inside2"] == 1.0 and s["coverage95"] == 1.0
    assert s["err_over_eps"] == pytest.approx(np.max(0.001 / (0.05 * TRUTH)))
    assert np.allclose(s["z"], 0.001 / 0.005)


def test_claimed_but_wrong_is_a_false_claim():
    s = score(TRUTH + 0.1, np.full(100, 0.005), TRUTH, rel_tol=0.05)  # sigma/eps small, error 20 sigma
    assert s["claimed"] and not s["success"] and s["false_claim"]
    assert s["inside2"] == 0.0 and s["coverage95"] == 0.0


def test_not_claimed_when_sigma_exceeds_tolerance_somewhere():
    sig = np.full(100, 0.005)
    sig[7] = 0.2  # eps_hat there is 0.05 * ~1.07
    s = score(TRUTH, sig, TRUTH, rel_tol=0.05)
    assert not s["claimed"] and not s["success"] and not s["false_claim"]
    assert s["max_sigma_over_eps"] > 1.0


def test_coverage_threshold_is_95_percent_of_points():
    sig = np.full(100, 0.005)
    m = TRUTH.copy()
    m[:5] += 0.011  # 5 points at 2.2 sigma: inside2 = 0.95 exactly -> still a success
    assert score(m, sig, TRUTH, 0.05)["success"]
    m[:6] += 0.011
    assert not score(m, sig, TRUTH, 0.05)["success"]


def test_no_estimate_is_never_a_claim():
    s = score(None, None, TRUTH, 0.05)
    assert not s["has_estimate"] and not s["claimed"] and not s["success"]
    assert not score(np.full(100, np.nan), np.ones(100), TRUTH, 0.05)["has_estimate"]
