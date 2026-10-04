"""Diagnostics against ArviZ (arviz_stats), which implements Vehtari et al. (arXiv 1903.08008)."""

import arviz_stats as azs
import numpy as np
import pytest

from gcbml.mcmc import diagnostics as dg


def ar1(rho, n_chains=4, n_draws=1000, seed=0, shift=None):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_chains, n_draws))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    s = np.sqrt(1 - rho**2)
    for t in range(1, n_draws):
        x[:, t] = rho * x[:, t - 1] + s * e[:, t]
    if shift is not None:
        x = x + np.asarray(shift)[:, None]
    return x


RHOS = [0.0, 0.5, 0.9, 0.98, -0.5]


@pytest.mark.parametrize("rho", RHOS)
def test_split_rhat_matches_arviz(rho):
    x = ar1(rho, seed=1)
    assert dg.split_rhat(x) == pytest.approx(float(azs.rhat(x, method="split")), rel=1e-6)


@pytest.mark.parametrize("rho", RHOS)
def test_rhat_matches_arviz(rho):
    x = ar1(rho, seed=2)
    assert dg.rhat(x) == pytest.approx(float(azs.rhat(x, method="rank")), rel=1e-6)


@pytest.mark.parametrize("rho", RHOS)
def test_bulk_ess_matches_arviz(rho):
    x = ar1(rho, seed=3)
    assert dg.bulk_ess(x) == pytest.approx(float(azs.ess(x, method="bulk")), rel=1e-6)


@pytest.mark.parametrize("rho", RHOS)
def test_tail_ess_matches_arviz(rho):
    x = ar1(rho, seed=4)
    assert dg.tail_ess(x) == pytest.approx(float(azs.ess(x, method="tail", prob=(0.05, 0.95))), rel=1e-6)


def test_odd_number_of_draws_and_heavy_tails_match_arviz():
    rng = np.random.default_rng(5)
    x = rng.standard_cauchy((3, 501))
    assert dg.rhat(x) == pytest.approx(float(azs.rhat(x, method="rank")), rel=1e-6)
    assert dg.bulk_ess(x) == pytest.approx(float(azs.ess(x, method="bulk")), rel=1e-6)
    assert dg.tail_ess(x) == pytest.approx(float(azs.ess(x, method="tail", prob=(0.05, 0.95))), rel=1e-6)


def test_rank_normalize_matches_blom_formula():
    from scipy.stats import norm, rankdata

    x = np.random.default_rng(6).standard_normal((4, 50))
    r = rankdata(x.ravel()).reshape(x.shape)
    expected = norm.ppf((r - 3 / 8) / (x.size + 1 / 4))
    np.testing.assert_allclose(dg.rank_normalize(x), expected, rtol=1e-12)


def test_ess_of_iid_draws_is_near_number_of_draws():
    x = ar1(0.0, n_chains=4, n_draws=2000, seed=7)
    assert 0.7 * x.size < dg.bulk_ess(x) < 1.3 * x.size


def test_ess_of_ar1_is_near_theory():
    rho = 0.9
    x = ar1(rho, n_chains=4, n_draws=5000, seed=8)
    theory = x.size * (1 - rho) / (1 + rho)
    assert dg.bulk_ess(x) == pytest.approx(theory, rel=0.3)


def test_rhat_flags_chains_with_different_means():
    x = ar1(0.3, n_chains=2, n_draws=1000, seed=9, shift=[0.0, 1.0])
    assert dg.rhat(x) > 1.05
    assert dg.rhat(ar1(0.3, n_chains=4, seed=9)) < 1.01


def test_rhat_flags_chains_with_different_scales_at_same_location():
    x = ar1(0.3, seed=10)
    x[0] *= 0.2
    assert dg.rhat(x) > 1.01  # the folded R-hat catches it, plain split R-hat does not
    assert dg.rhat(x) > dg.split_rhat(x)


def test_summary_and_converged_apply_thresholds():
    good = ar1(0.2, n_chains=4, n_draws=1000, seed=11)
    bad = ar1(0.3, n_chains=2, n_draws=1000, seed=12, shift=[0.0, 1.0])
    summ = dg.summary({"a": good, "b": good + 5.0})
    assert set(summ) == {"a", "b"}
    assert summ["a"] == pytest.approx((dg.rhat(good), dg.bulk_ess(good), dg.tail_ess(good)))
    assert dg.converged(summ)
    assert not dg.converged({**summ, "c": (1.02, 1000.0, 1000.0)})
    assert not dg.converged({"c": (1.005, 399.0, 1000.0)})
    assert not dg.converged({"c": (1.005, 1000.0, 399.0)})
    assert dg.converged({"c": (1.005, 401.0, 401.0)})
    assert dg.converged({"c": (1.005, 1000.0, 1000.0)}, rhat_max=1.01, ess_min=400.0)
    assert not dg.converged(dg.summary({"c": bad}))
