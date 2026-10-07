"""Tests of the pre-asymptotic generator: b(hbar) = hbar^p / (1 + (hbar/h_s)^m)^(p/m)."""

import numpy as np
import pytest

from gcbml.synthetic import PreasymptoticTruth, preasymptotic_b, preasymptotic_truth


def test_b_zero_at_zero():
    assert preasymptotic_b(0.0, p=1.5, h_s=0.3, m=4.0) == 0.0


def test_b_asymptotic_power_law():
    p, h_s, m = 1.5, 0.3, 4.0
    h1, h2 = 1e-4, 1e-5
    ratio = preasymptotic_b(h1, p, h_s, m) / preasymptotic_b(h2, p, h_s, m)
    assert ratio == pytest.approx((h1 / h2) ** p, rel=1e-6)


def test_b_saturates_for_large_h():
    p, h_s, m = 1.5, 0.3, 4.0
    assert preasymptotic_b(1e4, p, h_s, m) == pytest.approx(h_s**p, rel=1e-6)


def test_b_infinite_h_s_is_pure_power_law():
    h = np.array([0.0, 0.1, 0.5, 1.0, 3.0])
    assert np.allclose(preasymptotic_b(h, 1.7, np.inf, 4.0), h**1.7)


def test_truth_structure():
    t = preasymptotic_truth(3)
    assert isinstance(t, PreasymptoticTruth)
    x = np.linspace(0, 1, 9)
    assert np.allclose(t.z(x, 0.0), t.z0(x))
    assert np.allclose(t.z0(x), t.mu0 + t.mu1 * x**2)
    hbar = 0.25
    b = preasymptotic_b(hbar, t.p, t.h_s, t.m)
    assert np.allclose(t.z(x, hbar), t.z0(x) - t.a0 * (1 + t.a1 * x**2) * b)


def test_reproducible_and_seed_dependent():
    a, b, c = preasymptotic_truth(7), preasymptotic_truth(7), preasymptotic_truth(8)
    assert a == b
    assert a != c


def test_control_mode_has_infinite_h_s():
    t = preasymptotic_truth(1, h_s_range=None)
    assert np.isinf(t.h_s)


def test_observe_is_reproducible_with_noise_sd():
    t = preasymptotic_truth(2)
    x = np.linspace(0, 1, 9)
    h = np.array([1.0, 0.5, 0.25])
    y1 = t.observe(x, h, 0.015, seed=5)
    y2 = t.observe(x, h, 0.015, seed=5)
    assert y1.shape == (3, 9)
    assert np.array_equal(y1, y2)
    big = t.observe(x, h, 0.015, seed=6) - np.array([t.z(x, hh) for hh in h])
    assert 0.005 < big.std() < 0.04
