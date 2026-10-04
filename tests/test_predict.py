import jax
import numpy as np
import pytest
from scipy import stats

import gcbml  # noqa: F401
from gcbml import predict, transforms

Z84 = stats.norm.ppf(0.84)
Z16 = stats.norm.ppf(0.16)
IDENT = transforms.get("identity")
LOG = transforms.get("log")
RECIP = transforms.get("reciprocal")


def uniform_w(S):
    return np.full(S, 1.0 / S)


# ---------------------------- weighted_quantile -------------------------------------------------


@pytest.mark.parametrize("S", [7, 50, 501])
def test_weighted_quantile_equal_weights_equals_numpy_hazen(S):
    rng = np.random.default_rng(S)
    x = rng.normal(size=(S, 4))
    for q in [0.0, 0.01, 0.16, 0.5, 0.84, 0.975, 1.0]:
        got = np.asarray(predict.weighted_quantile(x, uniform_w(S), q))
        np.testing.assert_allclose(got, np.quantile(x, q, axis=0, method="hazen"), rtol=1e-12, atol=1e-12)


def test_weighted_quantile_is_invariant_to_draw_order():
    rng = np.random.default_rng(0)
    x, w = rng.normal(size=(40, 2)), rng.uniform(size=40)
    w /= w.sum()
    perm = rng.permutation(40)
    np.testing.assert_allclose(
        predict.weighted_quantile(x, w, 0.3), predict.weighted_quantile(x[perm], w[perm], 0.3), rtol=1e-12
    )


def test_weighted_quantile_with_weights_matches_repeated_sample():
    rng = np.random.default_rng(1)
    S = 2000
    x = np.sort(rng.normal(size=S))
    m = rng.integers(1, 6, size=S)
    w = m / m.sum()
    rep = np.repeat(x, m)
    # exact at the centre of every atom: q_i = cumsum(w)_i - w_i / 2 is the median copy of draw i
    c = np.cumsum(w) - w / 2
    for i in [3, 400, 1000, 1700]:
        got = float(predict.weighted_quantile(x[:, None], w, c[i])[0])
        assert abs(got - np.quantile(rep, c[i], method="hazen")) < 1e-12
        assert abs(got - x[i]) < 1e-12
    # elsewhere the repeated sample is flat inside an atom and the weighted CDF interpolates between atom
    # centres: they differ by less than the gap between neighbouring draws (O(1/S))
    for q in [0.025, 0.16, 0.5, 0.84, 0.975]:
        got = float(predict.weighted_quantile(x[:, None], w, q)[0])
        assert abs(got - np.quantile(rep, q, method="hazen")) < 0.01


def test_weighted_quantile_hand_value_and_zero_weight_draws_do_not_count():
    # three draws 0, 1, 3 with weights 0.5, 0.25, 0.25: CDF knots at 0.25, 0.625, 0.875
    x = np.array([[0.0], [1.0], [3.0]])
    w = np.array([0.5, 0.25, 0.25])
    assert abs(float(predict.weighted_quantile(x, w, 0.25)[0]) - 0.0) < 1e-14
    assert abs(float(predict.weighted_quantile(x, w, 0.625)[0]) - 1.0) < 1e-14
    # halfway between the knots 0.25 and 0.625 -> 0.5
    assert abs(float(predict.weighted_quantile(x, w, 0.4375)[0]) - 0.5) < 1e-14
    assert abs(float(predict.weighted_quantile(x, w, 0.99)[0]) - 3.0) < 1e-14  # clamped
    x2 = np.array([[0.0], [100.0], [1.0], [3.0]])
    w2 = np.array([0.5, 0.0, 0.25, 0.25])
    assert abs(float(predict.weighted_quantile(x2, w2, 0.99)[0]) - 3.0) < 1e-14
    assert abs(float(predict.weighted_quantile(x2, w2, 0.4375)[0]) - 0.5) < 1e-14


# ---------------------------- sampling and summarize --------------------------------------------


def test_sample_gaussian_draws_layout_weights_and_moments():
    S, s, n_rep = 5, 3, 4000
    mean = np.arange(S * s, dtype=float).reshape(S, s)
    var = np.full((S, s), 0.25)
    w = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    out, wr = predict.sample_gaussian_draws(jax.random.key(0), mean, var, n_rep, w)
    out, wr = np.asarray(out), np.asarray(wr)
    assert out.shape == (S * n_rep, s) and wr.shape == (S * n_rep,)
    np.testing.assert_allclose(wr, np.repeat(w / n_rep, n_rep), rtol=1e-14)
    assert abs(wr.sum() - 1.0) < 1e-12
    blocks = out.reshape(S, n_rep, s)  # row k * n_rep + r belongs to draw k
    np.testing.assert_allclose(blocks.mean(axis=1), mean, atol=5 * 0.5 / np.sqrt(n_rep))
    np.testing.assert_allclose(blocks.std(axis=1), 0.5, rtol=0.05)
    zero, _ = predict.sample_gaussian_draws(jax.random.key(1), mean, np.zeros((S, s)), 3)
    np.testing.assert_array_equal(np.asarray(zero).reshape(S, 3, s), np.repeat(mean[:, None, :], 3, axis=1))
    _, w_default = predict.sample_gaussian_draws(jax.random.key(1), mean, var, 2)
    np.testing.assert_allclose(w_default, 1.0 / (2 * S))


def test_summarize_identity_sigma_equals_sd():
    z = np.random.default_rng(0).normal(2.0, 0.5, size=(20001, 2))
    out = predict.summarize(z, uniform_w(20001), IDENT)
    np.testing.assert_allclose(out["sigma"], 0.5, rtol=0.03)
    np.testing.assert_allclose(out["m"], 2.0, atol=0.02)
    np.testing.assert_allclose(out["q975"] - out["q025"], 2 * 1.959964 * 0.5, rtol=0.03)
    np.testing.assert_allclose(out["sigma"], (out["q84"] - out["q16"]) / 2, rtol=1e-14)
    assert set(out) == {"m", "sigma", "q025", "q975", "q16", "q84"}


def test_summarize_median_commutes_with_monotone_transforms():
    z = np.random.default_rng(1).normal(1.0, 0.4, size=(2001, 3))  # odd S: the median is a draw
    w = uniform_w(2001)
    for tr in (LOG, RECIP):
        zz = z if tr is LOG else np.exp(z)  # reciprocal needs z > 0
        out = predict.summarize(zz, w, tr)
        np.testing.assert_allclose(out["m"], tr.inverse(np.median(zz, axis=0)), rtol=1e-12)
        assert np.all(out["q16"] <= out["m"]) and np.all(out["m"] <= out["q84"])
    # even S, unequal weights: still within one draw spacing
    z2 = z[:2000]
    w2 = np.random.default_rng(2).uniform(0.5, 1.5, size=2000)
    w2 /= w2.sum()
    out = predict.summarize(z2, w2, LOG)
    ref = np.exp(predict.weighted_quantile(z2, w2, 0.5))
    np.testing.assert_allclose(out["m"], ref, rtol=2e-3)


# ---------------------------- sigma_env ---------------------------------------------------------


def test_sigma_env_hand_computation_and_weighting():
    c0 = np.array([[0.1, 0.2], [0.0, -0.1]])
    c1 = np.array([[0.3, 0.4], [0.2, 0.0]])
    P = np.array([[[1.0, 2.0]], [[1.5, 1.0]]])  # (S, s, k)
    sd = np.array([[0.5, 0.6], [0.2, 0.3]])
    kx = np.array([[[1.0, 0.8]], [[0.9, 1.0]]])
    mu = np.array([[2.0], [1.0]])
    Hb = np.array([[0.5, 0.25]])
    w = np.array([0.25, 0.75])

    def one(s):  # by hand: sum_j hbar^{2 p} ((c0 + c1 mu)^2 + sd^2 kx)
        tot = 0.0
        for j in range(2):
            tot += Hb[0, j] ** (2 * P[s, 0, j]) * (
                (c0[s, j] + c1[s, j] * mu[s, 0]) ** 2 + sd[s, j] ** 2 * kx[s, 0, j]
            )
        return tot

    assert (
        abs(one(0) - (0.25 * (0.49 + 0.25) + 0.25**4 * (1.0 + 0.36 * 0.8))) < 1e-15
    )  # the 0.1900... by hand
    got = predict.sigma_env(c0, c1, P, sd, kx, mu, Hb, w)
    np.testing.assert_allclose(got, [np.sqrt(0.25 * one(0) + 0.75 * one(1))], rtol=1e-13)


def test_sigma_env_zero_at_hbar_zero_and_monotone_in_hbar():
    rng = np.random.default_rng(3)
    S, s, k = 30, 4, 3
    c0, c1 = rng.normal(size=(S, k)), rng.normal(size=(S, k))
    P = rng.uniform(0.5, 3.0, size=(S, s, k))
    sd = rng.uniform(0.1, 1.0, size=(S, k))
    kx = rng.uniform(0.5, 1.0, size=(S, s, k))
    mu = rng.normal(size=(S, s))
    w = rng.uniform(size=S)
    w /= w.sum()
    zero = predict.sigma_env(c0, c1, P, sd, kx, mu, np.zeros((s, k)), w)
    np.testing.assert_array_equal(np.asarray(zero), 0.0)
    H1 = rng.uniform(0.0, 1.0, size=(s, k))
    H1[0, 0] = 0.0  # a component exactly at zero
    prev = np.asarray(predict.sigma_env(c0, c1, P, sd, kx, mu, H1, w))
    for _ in range(20):  # step componentwise upwards; sigma_env must not decrease
        H1 = H1 + rng.uniform(0.0, 0.2, size=(s, k)) * (rng.uniform(size=(s, k)) < 0.5)
        cur = np.asarray(predict.sigma_env(c0, c1, P, sd, kx, mu, H1, w))
        assert np.all(cur >= prev - 1e-14)
        prev = cur
    assert np.all(prev > 0)


# ---------------------------- fidelity / knowledge / aleatoric / total --------------------------


def test_sigma_fid_hand_computation():
    fl = np.array([[1.0], [3.0]])
    f0 = np.array([[0.0], [1.0]])
    w = np.array([0.5, 0.5])
    np.testing.assert_allclose(predict.sigma_fid(fl, f0, w, IDENT), [np.sqrt(0.5 * 1 + 0.5 * 4)], rtol=1e-14)
    # log transform: Lambda^{-1} = exp; draws (log 2, log 1) and (log 3, log 1) -> diffs 1 and 2
    fl = np.log([[2.0], [3.0]])
    f0 = np.log([[1.0], [1.0]])
    np.testing.assert_allclose(
        predict.sigma_fid(fl, f0, np.array([0.25, 0.75]), LOG), [np.sqrt(0.25 + 3.0)], rtol=1e-13
    )
    # same draws -> 0
    assert float(predict.sigma_fid(fl, fl, np.array([0.5, 0.5]), LOG)[0]) == 0.0


def test_sigma_know_is_zero_for_identical_draws_and_sd_for_gaussian():
    same = np.tile(np.array([[1.0, 2.0]]), (50, 1))
    np.testing.assert_array_equal(np.asarray(predict.sigma_know(same, uniform_w(50), IDENT)), 0.0)
    np.testing.assert_array_equal(np.asarray(predict.sigma_know(same, uniform_w(50), LOG)), 0.0)
    z = np.random.default_rng(0).normal(0.0, 0.3, size=(20001, 1))
    np.testing.assert_allclose(predict.sigma_know(z, uniform_w(20001), IDENT), 0.3, rtol=0.03)
    # log transform of N(0, 0.3^2): half-width (exp(0.3 z84) - exp(0.3 z16)) / 2 (z16 = -z84)
    np.testing.assert_allclose(
        predict.sigma_know(z, uniform_w(20001), LOG), (np.exp(0.3 * Z84) - np.exp(0.3 * Z16)) / 2, rtol=0.03
    )


def test_s0_zero_noise_is_zero_and_known_widths():
    S, s = 40, 2
    my = np.array([3.0, 5.0])
    out = predict.s0(my, np.zeros((S, s)), uniform_w(S), IDENT, jax.random.key(0), 10)
    np.testing.assert_array_equal(np.asarray(out), 0.0)
    out = predict.s0(my, np.zeros((S, s)), uniform_w(S), LOG, jax.random.key(0), 10)
    np.testing.assert_array_equal(np.asarray(out), 0.0)
    # identity: half-width of N(m, 0.7^2) is 0.7 (z84 - z16) / 2
    sd = np.full((200, s), 0.7)
    out = predict.s0(my, sd, uniform_w(200), IDENT, jax.random.key(1), 400)
    np.testing.assert_allclose(out, 0.7 * (Z84 - Z16) / 2, rtol=0.02)
    # log: y = m exp(e), half-width m (exp(0.7 z84) - exp(0.7 z16)) / 2, here on the physical scale
    out = predict.s0(my, sd, uniform_w(200), LOG, jax.random.key(2), 400)
    np.testing.assert_allclose(out, my * (np.exp(0.7 * Z84) - np.exp(0.7 * Z16)) / 2, rtol=0.02)


def test_sigma_tot_zero_noise_equals_sigma_epi_and_known_width():
    rng = np.random.default_rng(4)
    S = 800
    mu = rng.normal(1.0, 0.2, size=(S, 2))
    w = uniform_w(S)
    epi = np.asarray(predict.summarize(mu, w, IDENT)["sigma"])
    tot0 = predict.sigma_tot(mu, np.zeros((S, 2)), w, IDENT, jax.random.key(0), 5)
    np.testing.assert_allclose(tot0, epi, rtol=5e-3)  # equal up to the interpolation of repeated draws
    # all mu draws equal and noise sd 0.5: sigma_tot = 0.5 (z84 - z16) / 2 on the identity scale
    same = np.full((200, 1), 1.0)
    out = predict.sigma_tot(same, np.full((200, 1), 0.5), uniform_w(200), IDENT, jax.random.key(1), 500)
    np.testing.assert_allclose(out, 0.5 * (Z84 - Z16) / 2, rtol=0.02)
    # not a quadrature sum: here mu ~ N(0, 0.3^2) and e ~ N(0, 0.4^2) on the log scale,
    # so mu + e ~ N(0, 0.5^2) and sigma_tot is the half-width of exp of that
    mu = np.random.default_rng(5).normal(0.0, 0.3, size=(4000, 1))
    out = predict.sigma_tot(mu, np.full((4000, 1), 0.4), uniform_w(4000), LOG, jax.random.key(2), 100)
    np.testing.assert_allclose(out, (np.exp(0.5 * Z84) - np.exp(0.5 * Z16)) / 2, rtol=0.03)
