import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import gcbml  # noqa: F401
from gcbml import cost
from gcbml._config import bucket
from gcbml.cost import CostData, CostPosterior, CostPrior

# ----------------------------------------------------------------------------------------------
# Independent NumPy reference and simulator, written from the cost.py docstring (spec 2.6)
# ----------------------------------------------------------------------------------------------


def np_matern52(r):
    return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)


def np_ard52(X1, X2, ell):
    d = (X1[:, None, :] - X2[None, :, :]) / ell
    return np_matern52(np.sqrt(np.sum(d**2, axis=-1)))


def np_inputs(U, L, l_scale):
    return np.concatenate([U, L / l_scale], axis=1)


def pad_data(U, L, y, censored, log2q, has_q, n_pad, garbage=None):
    """Pad real rows to n_pad. ``garbage`` (a Generator) fills the padded rows with junk instead of zeros."""
    n, d = U.shape

    def pad(a, junk):
        out = np.zeros((n_pad,) + a.shape[1:], dtype=a.dtype)
        out[:n] = a
        if garbage is not None:
            out[n:] = junk(out[n:].shape)
        return out

    g = garbage
    return CostData(
        U=jnp.asarray(pad(U, (lambda s: g.normal(size=s) * 50) if g else None)),
        L=jnp.asarray(pad(L, (lambda s: g.normal(size=s) * 50) if g else None)),
        log2c=jnp.asarray(pad(y, (lambda s: g.normal(size=s) * 50) if g else None)),
        censored=jnp.asarray(pad(censored, (lambda s: g.random(s) < 0.5) if g else None)),
        log2q=jnp.asarray(pad(log2q, (lambda s: g.normal(size=s) * 50) if g else None)),
        has_q=jnp.asarray(pad(has_q, (lambda s: g.random(s) < 0.5) if g else None)),
        mask=jnp.asarray(np.arange(n_pad) < n),
    )


def simulate(
    rng,
    n,
    k,
    d_u=1,
    kappa0=5.0,
    gamma=(3.0,),
    a=0.0,
    sw=0.5,
    ell=None,
    s_eta=0.15,
    l_max=3,
    q_frac=0.0,
    l_scale=4.0,
    log2q_fn=None,
    omega_fn=None,
):
    """Draw n runs from the model of the cost.py docstring. Returns dict of unpadded arrays and truth."""
    U = rng.random((n, d_u))
    L = rng.integers(0, l_max + 1, size=(n, k)).astype(float)
    gamma = np.asarray(gamma, dtype=float)
    ell = np.full(d_u + k, 0.7) if ell is None else np.asarray(ell, dtype=float)
    X = np_inputs(U, L, l_scale)
    if omega_fn is None:
        K = sw**2 * np_ard52(X, X, ell) + 1e-10 * np.eye(n)
        omega = np.linalg.cholesky(K) @ rng.standard_normal(n)
    else:
        omega = omega_fn(U, L)
    eta = s_eta * rng.standard_normal(n)
    has_q = rng.random(n) < q_frac
    clean = kappa0 + L @ gamma + omega
    if log2q_fn is None:
        log2q = np.where(has_q, rng.normal(8.0, 1.5, n), 0.0)
        qbar = log2q[has_q].mean() if has_q.any() else 0.0
        y = clean + a * has_q * (log2q - qbar) + eta
    else:
        y = clean + eta
        log2q = np.where(has_q, log2q_fn(y, rng), 0.0)
    return dict(U=U, L=L, y=y, has_q=has_q, log2q=log2q, clean=clean)


def make_data(sim, n_pad=None, cap=None, **kw):
    n = len(sim["y"])
    y, cens = sim["y"].copy(), np.zeros(n, dtype=bool)
    if cap is not None:
        cens = y > cap
        y = np.where(cens, cap, y)
    return pad_data(sim["U"], sim["L"], y, cens, sim["log2q"], sim["has_q"], n_pad or bucket(n), **kw)


def sample_coef(post, rng, n=4000):
    """Pooled draws of (kappa0, gamma, a): beta | y, hyper is Gaussian for every hyper draw."""
    mean, cov = (np.asarray(v) for v in cost.coef_posterior(post))
    S = mean.shape[0]
    idx = rng.integers(0, S, size=n)
    out = np.empty((n, mean.shape[1]))
    for t, s in enumerate(idx):
        out[t] = rng.multivariate_normal(mean[s], cov[s])
    return out


def mixture_moments(mean, var, w=None):
    mean, var = np.asarray(mean), np.asarray(var)
    w = np.full(mean.shape[0], 1.0 / mean.shape[0]) if w is None else np.asarray(w) / np.sum(w)
    m = w @ mean
    return m, w @ (var + mean**2) - m**2


FAST = dict(n_warmup=150, n_samples=150, n_chains=4)

# ----------------------------------------------------------------------------------------------
# 1. exact integration
# ----------------------------------------------------------------------------------------------


def test_integrated_predictive_equals_brute_force_joint_gaussian():
    rng = np.random.default_rng(0)
    n, m, d_u, k, n_pad = 12, 5, 2, 2, 16
    prior = CostPrior(k0_mean=1.0, k0_sd=2.0, gamma_mean=(3.0, 2.0), gamma_sd=(1.0, 1.5), l_scale=4.0)
    sim = simulate(rng, n, k, d_u=d_u, gamma=(3.0, 2.0), a=0.5, q_frac=0.5)
    data = make_data(sim, n_pad)
    # two hyperparameter draws and two different imputed-y draws
    sigma_w, s_eta = np.array([0.7, 1.3]), np.array([0.2, 0.5])
    ell = np.array([[0.5, 0.8, 1.1, 0.9], [1.0, 0.6, 0.7, 1.5]])
    y_pad = np.zeros((2, n_pad))
    y_pad[0, :n] = sim["y"]
    y_pad[1, :n] = sim["y"] + rng.normal(0, 0.3, n)
    post = CostPosterior(
        hyper=dict(sigma_w=jnp.asarray(sigma_w), s_eta=jnp.asarray(s_eta), ell=jnp.asarray(ell)),
        log2c=jnp.asarray(y_pad),
        diagnostics={},
        data=data,
        prior=prior,
    )
    Un, Ln = rng.random((m, d_u)), rng.integers(0, 4, size=(m, k)).astype(float)
    hq_n = np.array([True, False, True, False, False])
    lq_n = np.where(hq_n, rng.normal(8, 1.5, m), 0.0)
    mean, var = cost.predict_log2(post, Un, Ln, lq_n, hq_n)
    mean, var = np.asarray(mean), np.asarray(var)
    assert mean.shape == (2, m) and var.shape == (2, m)

    # brute force: latent vector (kappa0, gamma, a, omega over n + m runs, eta over n + m runs)
    qbar = sim["log2q"][sim["has_q"]].mean()
    q = 2 + k
    b0 = np.array([1.0, 3.0, 2.0, 1.0])
    B = np.diag([2.0, 1.0, 1.5, 0.5]) ** 2
    Uall, Lall = np.vstack([sim["U"], Un]), np.vstack([sim["L"], Ln])
    hq_all = np.concatenate([sim["has_q"], hq_n])
    lq_all = np.concatenate([sim["log2q"], lq_n])
    A = np.column_stack([np.ones(n + m), Lall, hq_all * (lq_all - qbar)])
    X = np_inputs(Uall, Lall, 4.0)
    for s in range(2):
        N = n + m
        Sigma = np.zeros((q + 2 * N, q + 2 * N))
        Sigma[:q, :q] = B
        Sigma[q : q + N, q : q + N] = sigma_w[s] ** 2 * np_ard52(X, X, ell[s])
        Sigma[q + N :, q + N :] = s_eta[s] ** 2 * np.eye(N)
        Phi = np.hstack([A, np.eye(N), np.eye(N)])
        C = Phi @ Sigma @ Phi.T
        mu = A @ b0
        Coo, Cno, Cnn = C[:n, :n], C[n:, :n], C[n:, n:]
        w = np.linalg.solve(Coo, np.column_stack([y_pad[s, :n] - mu[:n], Cno.T]))
        ref_mean = mu[n:] + Cno @ w[:, 0]
        ref_var = np.diag(Cnn) - np.sum(Cno * w[:, 1:].T, axis=1)
        np.testing.assert_allclose(mean[s], ref_mean, rtol=0, atol=1e-8)
        np.testing.assert_allclose(var[s], ref_var, rtol=0, atol=1e-8)


# ----------------------------------------------------------------------------------------------
# 2. recovery and calibration
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2])
def test_posterior_covers_true_gamma(k):
    rng = np.random.default_rng(10 + k)
    gamma = (3.0,) if k == 1 else (3.0, 2.5)
    sim = simulate(rng, 60, k, gamma=gamma, a=0.6, q_frac=0.6, sw=0.5, s_eta=0.15)
    prior = CostPrior(4.0, 3.0, (2.5,) * k, (1.0,) * k)
    post = cost.fit_cost(jax.random.key(k), make_data(sim), prior, **FAST)
    draws = sample_coef(post, rng)
    lo, hi = np.percentile(draws, [0.5, 99.5], axis=0)
    truth = np.concatenate([[5.0], gamma, [0.6]])
    assert np.all((lo <= truth) & (truth <= hi)), (lo, hi, truth)
    # the posterior is informative: much tighter than the gamma prior (sd 1)
    assert np.all(draws[:, 1 : 1 + k].std(axis=0) < 0.35)


@pytest.mark.slow
def test_predictive_is_calibrated_at_held_out_runs():
    zs = []
    for rep in range(6):
        rng = np.random.default_rng(100 + rep)
        sim = simulate(rng, 140, 2, d_u=2, gamma=(3.0, 2.0), a=0.6, q_frac=0.5, sw=0.6, s_eta=0.2)
        tr = {kk: v[:60] for kk, v in sim.items()}
        te = {kk: v[60:] for kk, v in sim.items()}
        # qbar is a property of the training quotes: held-out y used the pooled qbar of the simulator,
        # so re-centre the held-out quote effect on the training qbar
        qbar_all = sim["log2q"][sim["has_q"]].mean()
        qbar_tr = tr["log2q"][tr["has_q"]].mean()
        te["y"] = te["y"] + 0.6 * te["has_q"] * (qbar_all - qbar_tr)
        prior = CostPrior(4.0, 3.0, (2.5, 2.5), (1.0, 1.0))
        post = cost.fit_cost(jax.random.key(rep), make_data(tr), prior, 300, 300, 4)
        mean, var = cost.predict_log2(post, te["U"], te["L"], te["log2q"], te["has_q"])
        mu, v = mixture_moments(mean, var)
        zs.append((te["y"] - mu) / np.sqrt(v))
    z = np.concatenate(zs)
    assert abs(z.mean()) < 0.2, z.mean()
    assert 0.8 < z.std() < 1.25, z.std()
    assert 0.90 < np.mean(np.abs(z) < 1.96) < 0.99


# ----------------------------------------------------------------------------------------------
# 3. expressiveness: omega absorbs a bend away from the straight line
# ----------------------------------------------------------------------------------------------


def test_omega_absorbs_a_bend_in_level_and_straight_line_is_biased():
    rng = np.random.default_rng(3)

    def bend(lv):  # gamma = 1 for l < 2, 5 beyond: 0, 1, 2, 7, 12, 17
        return 1.0 * np.minimum(lv, 2) + 5.0 * np.maximum(lv - 2, 0)

    n = 50
    U = rng.random((n, 1))
    L = rng.choice([0.0, 1.0, 2.0, 4.0, 5.0], size=(n, 1))
    y = bend(L[:, 0]) + 0.1 * rng.standard_normal(n)
    Un = rng.random((10, 1))
    Ln = np.full((10, 1), 3.0)
    truth = bend(3.0)
    prior = CostPrior(0.0, 3.0, (3.0,), (1.0,), l_scale=5.0)
    data = pad_data(U, L, y, np.zeros(n, bool), np.zeros(n), np.zeros(n, bool), bucket(n))
    post = cost.fit_cost(jax.random.key(0), data, prior, **FAST)
    mean, var = cost.predict_log2(post, Un, Ln, np.zeros(10), np.zeros(10, bool))
    mu, v = mixture_moments(mean, var)
    gp_err = np.abs(mu - truth).max()

    # straight-line-only model: same priors on (kappa0, gamma), no omega, noise by OLS residual
    A = np.column_stack([np.ones(n), L[:, 0]])
    b0, B = np.array([0.0, 3.0]), np.diag([3.0, 1.0]) ** 2
    s2 = np.var(y - A @ np.linalg.lstsq(A, y, rcond=None)[0])
    Binv = np.diag(1.0 / np.diag(B))
    beta = np.linalg.solve(A.T @ A / s2 + Binv, A.T @ y / s2 + Binv @ b0)
    line_err = np.abs(beta[0] + beta[1] * 3.0 - truth)

    assert line_err > 1.2, line_err  # the straight line is clearly biased at the interior level
    assert gp_err < 0.5, gp_err
    assert gp_err < 0.25 * line_err
    # and the calibrated predictive covers the truth
    assert np.all(np.abs(mu - truth) < 3 * np.sqrt(v))


# ----------------------------------------------------------------------------------------------
# 4. Tobit
# ----------------------------------------------------------------------------------------------


def test_tobit_gamma_consistent_with_uncensored_naive_is_biased_low():
    rng = np.random.default_rng(4)
    sim = simulate(rng, 60, 1, gamma=(3.0,), sw=0.3, s_eta=0.3, l_max=4)
    cap = np.quantile(sim["y"], 0.7)
    prior = CostPrior(5.0, 3.0, (3.0,), (1.0,))
    key = jax.random.key(4)

    def gamma_draws(data):
        return sample_coef(cost.fit_cost(key, data, prior, **FAST), np.random.default_rng(0))[:, 1]

    g_full = gamma_draws(make_data(sim))
    g_tobit = gamma_draws(make_data(sim, cap=cap))
    capped = make_data(sim, cap=cap)
    assert int(np.sum(np.asarray(capped.censored))) >= 12  # a real fraction is censored
    naive = capped._replace(censored=jnp.zeros_like(capped.censored))
    g_naive = gamma_draws(naive)

    gap_tobit = abs(g_tobit.mean() - g_full.mean())
    gap_naive = g_full.mean() - g_naive.mean()
    assert gap_tobit < 2 * g_full.std() + 0.05, (gap_tobit, g_full.std())
    assert gap_naive > 3 * g_full.std(), (gap_naive, g_full.std())  # biased low, clearly
    assert gap_tobit < 0.5 * gap_naive


def test_censored_imputations_respect_the_cap():
    rng = np.random.default_rng(5)
    sim = simulate(rng, 40, 1, gamma=(3.0,), s_eta=0.3)
    cap = np.quantile(sim["y"], 0.6)
    data = make_data(sim, cap=cap)
    post = cost.fit_cost(jax.random.key(5), data, CostPrior(5.0, 3.0, (3.0,), (1.0,)), 100, 100, 4)
    y = np.asarray(post.log2c)
    cens = np.asarray(data.censored)
    assert np.all(y[:, cens] >= cap - 1e-12)
    # uncensored rows are untouched
    obs = np.asarray(data.mask) & ~cens
    np.testing.assert_array_equal(y[:, obs], np.broadcast_to(np.asarray(data.log2c)[obs], y[:, obs].shape))
    # the imputations are not just pinned at the cap
    assert y[:, cens].mean() > cap + 0.05


# ----------------------------------------------------------------------------------------------
# 5. quotes
# ----------------------------------------------------------------------------------------------


def _fit_quote_case(rng, log2q_fn, q_frac_fit=1.0, seed=0):
    sim = simulate(rng, 60, 1, gamma=(3.0,), sw=0.3, s_eta=0.3, q_frac=q_frac_fit, log2q_fn=log2q_fn, l_max=3)
    prior = CostPrior(5.0, 3.0, (3.0,), (1.0,))
    return sim, prior


def test_informative_quotes_give_a_near_one_and_shrink_predictive_variance():
    rng = np.random.default_rng(6)

    def quote(y, rng):  # quote = true cost times lognormal noise (log2 sd 0.05)
        return y + 0.05 * rng.standard_normal(y.shape)

    sim, prior = _fit_quote_case(rng, quote)
    post_q = cost.fit_cost(jax.random.key(6), make_data(sim), prior, **FAST)
    a = sample_coef(post_q, rng)[:, -1]
    assert abs(a.mean() - 1.0) < 0.15 and a.std() < 0.2, (a.mean(), a.std())
    # same runs, quotes hidden
    hidden = make_data(sim)._replace(has_q=jnp.zeros_like(make_data(sim).has_q))
    post_n = cost.fit_cost(jax.random.key(6), hidden, prior, **FAST)
    te = simulate(rng, 30, 1, gamma=(3.0,), sw=0.3, s_eta=0.3, q_frac=1.0, log2q_fn=quote)
    mq, vq = cost.predict_log2(post_q, te["U"], te["L"], te["log2q"], te["has_q"])
    mn, vn = cost.predict_log2(post_n, te["U"], te["L"], np.zeros(30), np.zeros(30, bool))
    sd_q = np.sqrt(mixture_moments(mq, vq)[1])
    sd_n = np.sqrt(mixture_moments(mn, vn)[1])
    assert np.median(sd_q) < 0.5 * np.median(sd_n), (np.median(sd_q), np.median(sd_n))
    # and the quote predictions are right
    assert (
        np.abs(mixture_moments(mq, vq)[0] - te["y"]).mean()
        < np.abs(mixture_moments(mn, vn)[0] - te["y"]).mean()
    )


def test_pure_noise_quotes_give_a_near_zero():
    rng = np.random.default_rng(7)

    def quote(y, rng):
        return rng.normal(8.0, 2.0, y.shape)

    sim, prior = _fit_quote_case(rng, quote)
    post = cost.fit_cost(jax.random.key(7), make_data(sim), prior, **FAST)
    a = sample_coef(post, rng)[:, -1]
    assert abs(a.mean()) < 0.15, a.mean()
    assert np.percentile(a, 0.5) < 0 < np.percentile(a, 99.5)


# ----------------------------------------------------------------------------------------------
# 6. expected_cost and cost_cap
# ----------------------------------------------------------------------------------------------


def test_expected_cost_matches_monte_carlo_of_the_lognormal_mixture():
    rng = np.random.default_rng(8)
    S, m = 6, 3
    mean = rng.normal(4.0, 1.0, (S, m))
    var = rng.uniform(0.1, 0.8, (S, m))
    w = rng.random(S) + 0.2
    got = np.asarray(cost.expected_cost(mean, var, w))
    N = 2_000_000
    ref = np.empty(m)
    se = np.empty(m)
    for j in range(m):
        s = rng.choice(S, size=N, p=w / w.sum())
        c = 2.0 ** (mean[s, j] + np.sqrt(var[s, j]) * rng.standard_normal(N))
        ref[j], se[j] = c.mean(), c.std() / np.sqrt(N)
    assert np.all(np.abs(got - ref) < 4 * se), (got, ref, se)
    # closed form of one lognormal
    one = np.asarray(cost.expected_cost(mean[:1], var[:1], np.ones(1)))
    np.testing.assert_allclose(one, 2.0 ** mean[0] * np.exp(0.5 * np.log(2) ** 2 * var[0]), rtol=1e-12)


def test_cost_cap_is_the_mixture_quantile_and_single_draw_is_closed_form():
    rng = np.random.default_rng(9)
    S, m = 7, 4
    mean = rng.normal(4.0, 2.0, (S, m))
    var = rng.uniform(0.05, 3.0, (S, m))
    w = rng.random(S) + 0.1
    for q in (0.5, 0.95, 0.99):
        cap = np.asarray(cost.cost_cap(mean, var, w, q))
        lc = np.log2(cap)
        cdf = (w / w.sum()) @ stats.norm.cdf((lc - mean) / np.sqrt(var))
        np.testing.assert_allclose(cdf, q, rtol=0, atol=1e-6)
    one = np.asarray(cost.cost_cap(mean[:1], var[:1], np.ones(1), 0.95))
    np.testing.assert_allclose(one, 2.0 ** (mean[0] + stats.norm.ppf(0.95) * np.sqrt(var[0])), rtol=1e-9)
    # jittable
    jitted = jax.jit(cost.cost_cap, static_argnames="q")(mean, var, w, q=0.95)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(cost.cost_cap(mean, var, w, 0.95)), rtol=1e-12)


# ----------------------------------------------------------------------------------------------
# 7. padding
# ----------------------------------------------------------------------------------------------


def test_garbage_in_padded_rows_changes_nothing():
    rng = np.random.default_rng(11)
    sim = simulate(rng, 20, 1, gamma=(3.0,), a=0.5, q_frac=0.5, s_eta=0.3)
    cap = np.quantile(sim["y"], 0.8)
    prior = CostPrior(5.0, 3.0, (3.0,), (1.0,))
    clean = make_data(sim, 32, cap=cap)
    dirty = make_data(sim, 32, cap=cap, garbage=np.random.default_rng(99))
    dirty = dirty._replace(U=dirty.U.at[25:].set(jnp.nan), log2q=dirty.log2q.at[26:].set(jnp.inf))
    key = jax.random.key(11)
    p1 = cost.fit_cost(key, clean, prior, 60, 40, 2)
    p2 = cost.fit_cost(key, dirty, prior, 60, 40, 2)
    for name in p1.hyper:
        np.testing.assert_allclose(np.asarray(p1.hyper[name]), np.asarray(p2.hyper[name]), rtol=1e-12)
    n = 20
    np.testing.assert_allclose(np.asarray(p1.log2c)[:, :n], np.asarray(p2.log2c)[:, :n], rtol=1e-12)
    Un, Ln = rng.random((4, 1)), rng.integers(0, 4, (4, 1)).astype(float)
    args = (Un, Ln, np.full(4, 7.0), np.array([True, False, True, False]))
    m1, v1 = cost.predict_log2(p1, *args)
    m2, v2 = cost.predict_log2(p2, *args)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(m2), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(v1), np.asarray(v2), rtol=1e-12)
    # same real data, different bucket size: same predictive for fixed hyper and y
    big = make_data(sim, 48, cap=cap)
    p3 = p1._replace(data=big, log2c=jnp.pad(p1.log2c[:, :n], ((0, 0), (0, 48 - n))))
    m3, v3 = cost.predict_log2(p3, *args)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(m3), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(v1), np.asarray(v3), rtol=1e-9, atol=1e-9)


def test_fit_reports_converging_diagnostics_and_shapes():
    rng = np.random.default_rng(12)
    sim = simulate(rng, 40, 1, gamma=(3.0,), s_eta=0.2)
    post = cost.fit_cost(jax.random.key(12), make_data(sim), CostPrior(5.0, 3.0, (3.0,), (1.0,)), 100, 50, 4)
    S = 4 * 50
    assert post.hyper["sigma_w"].shape == (S,)
    assert post.hyper["s_eta"].shape == (S,)
    assert post.hyper["ell"].shape == (S, 2)
    assert post.log2c.shape == (S, bucket(40))
    assert {"sigma_w", "s_eta"} <= set(post.diagnostics)
    assert np.all(np.isfinite(post.diagnostics["sigma_w"]))
