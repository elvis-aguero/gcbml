"""Gibbs inference for one structure (spec 2.5): recovery, prior check, calibration, censoring, order.

Data are simulated in NumPy from the generative model of spec 2.2-2.4 (the simulator below is written from
the model.py docstring, not from model.py), then fitted with gcbml.inference.fit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import gcbml  # noqa: F401  (float64)
from gcbml import inference, model
from gcbml._config import bucket
from gcbml.data import PaddedData
from gcbml.mcmc import diagnostics as dg
from gcbml.model import ModelConfig
from gcbml.priors import PriorScales

# ----------------------------------------------------------------------------------------------
# NumPy simulator of the Yi-h generative model
# ----------------------------------------------------------------------------------------------


def np_matern(r, nu):
    if nu == 1.5:
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    if nu == 2.5:
        return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    raise ValueError(nu)


def np_ard(X1, X2, ell, nu):
    diff = (X1[:, None, :] - X2[None, :, :]) / ell
    return np_matern(np.sqrt(np.sum(diff**2, axis=-1)), nu)


def draw_gp(rng, K):
    """One draw of N(0, K) by a jittered Cholesky factor (K may be singular: replicated points)."""
    L = np.linalg.cholesky(K + 1e-10 * np.mean(np.diag(K)) * np.eye(len(K)))
    return L @ rng.standard_normal(len(K))


def simulate(rng, T, X, H, Xs, P=None, zeta_row=None, add_noise=True):
    """z = rho0 + rho1 mu + delta + e at the rows (X, H); also returns the true mu at the test points Xs.

    T: sigma_mu, ell_mu (d,), beta0, c0 (k,), c1 (k,), sigma_delta (k,), ell_x (k, d), ell_h (k,), p0 (k,),
    m_s, b_s (k,). P: (n, k) per-row orders (default: the constant p0). zeta_row: (n,) log-variance field.
    """
    n, k = H.shape
    Xa = np.vstack([X, Xs])
    mu_all = T["beta0"] + draw_gp(rng, T["sigma_mu"] ** 2 * np_ard(Xa, Xa, T["ell_mu"], 2.5))
    P = np.tile(T["p0"], (n, 1)) if P is None else P
    hp = np.where(H > 0, np.where(H > 0, H, 1.0) ** P, 0.0)
    Kd = np.zeros((n, n))
    for j in range(k):
        kh = (
            hp[:, j][:, None]
            * hp[:, j][None, :]
            * np_matern(np.abs(H[:, j][:, None] - H[:, j]) / T["ell_h"][j], 1.5)
        )
        Kd += T["sigma_delta"][j] ** 2 * np_ard(X, X, T["ell_x"][j], 2.5) * kh
    delta = draw_gp(rng, Kd)
    rho0, rho1 = hp @ T["c0"], 1.0 + hp @ T["c1"]
    log_s2 = T["m_s"] + H @ T["b_s"] + (0.0 if zeta_row is None else zeta_row)
    z = rho0 + rho1 * mu_all[:n] + delta + add_noise * np.exp(0.5 * log_s2) * rng.standard_normal(n)
    return z, mu_all[n:]


def pad(X, H, z, n_pad=None, censored=None, run=None):
    n = len(z)
    n_pad = bucket(n) if n_pad is None else n_pad
    p = n_pad - n
    cens = np.zeros(n, bool) if censored is None else censored
    data = PaddedData(
        X=np.vstack([X, np.repeat(X[:1], p, 0)]),
        H=np.vstack([H, np.repeat(H[:1], p, 0)]),
        y=np.concatenate([z, np.repeat(z[:1], p)]),
        censored=np.concatenate([cens, np.zeros(p, bool)]),
        run=np.concatenate([np.arange(n) if run is None else run, np.full(p, -1)]),
        mask=np.concatenate([np.ones(n, bool), np.zeros(p, bool)]),
    )
    return data, np.concatenate([z, np.repeat(z[:1], p)])


def design_1d(n_coarse=8, n_mid=6, n_fine=4, n_rep=3):
    """k = 1 resolution component, 3 levels hbar = 1, 1/2, 1/4, d = 1 control; replicates at coarse sites."""
    u = [np.linspace(0.05, 0.95, m) for m in (n_coarse, n_mid, n_fine)]
    h = [np.full(m, v) for m, v in zip((n_coarse, n_mid, n_fine), (1.0, 0.5, 0.25))]
    rep = u[0][:n_rep]
    X = np.concatenate(u + [rep])[:, None]
    H = np.concatenate(h + [np.full(n_rep, 1.0)])[:, None]
    return X, H


TRUTH = dict(
    sigma_mu=0.8,
    ell_mu=np.array([0.4]),
    beta0=0.5,
    c0=np.array([0.6]),
    c1=np.array([-0.3]),
    sigma_delta=np.array([0.3]),
    ell_x=np.array([[0.5]]),
    ell_h=np.array([0.5]),
    p0=np.array([1.5]),
    m_s=np.log(0.05**2),
    b_s=np.array([0.0]),
)
SCALES = PriorScales(S_mu=1.0, S_c=1.0, S_delta=0.5, S_noise=0.05)
XS = np.linspace(0.1, 0.9, 5)[:, None]


def predict_mu_draws(post, data, cfg, Xs, rng, thin=4):
    """Draws of mu(x*) given each (thinned) posterior draw: N(predict_mu mean, var)."""
    params, z = inference.flatten(post)
    idx = np.arange(0, z.shape[0], thin)
    sel = jax.tree_util.tree_map(lambda a: a[idx], params)
    mean, var = jax.jit(jax.vmap(lambda p, zz: model.predict_mu(p, data, zz, cfg, Xs)))(sel, z[idx])
    mean, var = np.asarray(mean), np.asarray(var)
    return mean + np.sqrt(var) * rng.standard_normal(mean.shape)


@pytest.fixture(scope="module")
def recovery():
    rng = np.random.default_rng(20260401)
    X, H = design_1d()
    z, mu_true = simulate(rng, TRUTH, X, H, XS)
    data, zp = pad(X, H, z)
    cfg = ModelConfig()
    post = inference.fit(
        jax.random.key(1), data, zp, zp, cfg, SCALES, n_controls=1, n_warmup=300, n_samples=300
    )
    return data, cfg, post, mu_true, (X, H, z)


def test_recovery_posterior_covers_true_mu_and_p0(recovery):
    data, cfg, post, mu_true, _ = recovery
    mu = predict_mu_draws(post, data, cfg, XS, np.random.default_rng(0))
    lo, hi = np.quantile(mu, [0.005, 0.995], axis=0)
    assert np.all((lo <= mu_true) & (mu_true <= hi)), (lo, mu_true, hi)
    lp0 = np.asarray(post.theta["log_p0"]).reshape(-1)
    lo, hi = np.quantile(lp0, [0.025, 0.975])
    assert lo <= np.log(TRUTH["p0"][0]) <= hi, (lo, np.log(1.5), hi)
    assert np.all(np.isfinite(lp0))


def test_flatten_and_diagnostics_shapes(recovery):
    data, cfg, post, _, _ = recovery
    n_c, n_s = 4, 300
    n_pad = data.X.shape[0]
    assert post.z.shape == (n_c, n_s, n_pad)
    assert post.params.c0.shape == (n_c, n_s, 1)
    assert post.params.ell_mu.shape == (n_c, n_s, 1)
    assert post.params.P.shape == (n_c, n_s, n_pad, 1)
    assert post.params.noise_var.shape == (n_c, n_s, n_pad)
    assert post.params.delta.ell_x.shape == (n_c, n_s, 1, 1)
    flat_params, flat_z = inference.flatten(post)
    assert flat_z.shape == (n_c * n_s, n_pad)
    assert flat_params.sigma_mu.shape == (n_c * n_s,)
    assert flat_params.P.shape == (n_c * n_s, n_pad, 1)
    # pooled draws are the chains laid end to end
    np.testing.assert_array_equal(np.asarray(flat_z[:n_s]), np.asarray(post.z[0]))
    np.testing.assert_array_equal(np.asarray(flat_z[n_s : 2 * n_s]), np.asarray(post.z[1]))
    # unconstrained draws and constrained draws agree
    np.testing.assert_allclose(np.exp(post.theta["log_sigma_mu"]), post.params.sigma_mu, rtol=1e-12)
    # one diagnostic triple per scalar of theta (and of the noise hyperparameters)
    n_scalars = sum(int(np.prod(v.shape[2:])) for v in post.theta.values())
    assert len(post.diagnostics) == n_scalars
    assert "log_sigma_z" in " ".join(post.diagnostics)
    for name, (rh, bulk, tail) in post.diagnostics.items():
        assert np.isfinite(rh) and np.isfinite(bulk) and np.isfinite(tail), name
    assert post.n_evals > 4 * 600  # at least one evaluation per iteration and chain


def test_gibbs_step_is_jitted_once_and_traces_the_likelihood_once_per_use(monkeypatch):
    rng = np.random.default_rng(3)
    X, H = design_1d(5, 4, 3, 2)
    z, _ = simulate(rng, TRUTH, X, H, XS)
    data, zp = pad(X, H, z)
    calls = {"n": 0}
    real = inference.log_marginal

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(inference, "log_marginal", counting)
    sampler = inference._Sampler(data, zp, zp, ModelConfig(), SCALES, 1, False)
    init = sampler.init_chains(jax.random.key(0), 2)
    step = jax.jit(jax.vmap(sampler.make_step(sampler.init_widths())))
    n_after_init = calls["n"]
    state = init
    for i in range(4):  # different keys, different states
        state, _ = step(jax.random.split(jax.random.key(i), 2), state)
    jax.block_until_ready(state)
    assert step._cache_size() == 1  # one compilation for all iterations
    n_traced = calls["n"] - n_after_init
    assert n_traced > 0
    state, _ = step(jax.random.split(jax.random.key(99), 2), state)
    assert calls["n"] - n_after_init == n_traced  # no retrace on the next call


def test_censored_outputs_respect_bounds_and_mu_overlaps_uncensored_fit(recovery):
    d_u, cfg, post_u, mu_true, (X, H, z) = recovery  # the uncensored fit of the same data
    n = len(z)
    cens_rows = np.array([0, 1, 9, 14])  # right-censored: only z >= bound is known
    cens = np.zeros(n, bool)
    cens[cens_rows] = True
    bounds = z.copy()
    bounds[cens] = z[cens] - 0.3
    z_init = np.where(cens, bounds, z)
    d_c, zc = pad(X, H, z_init, censored=cens)
    _, bc = pad(X, H, bounds)
    post_c = inference.fit(jax.random.key(2), d_c, zc, bc, cfg, SCALES, 1, n_warmup=300, n_samples=300)
    zdraw = np.asarray(post_c.z)
    assert np.all(zdraw[:, :, cens_rows] >= bounds[cens_rows] - 1e-12)
    # imputed values move away from the bound (the model pulls them towards the truth side)
    assert np.mean(zdraw[:, :, cens_rows] - bounds[cens_rows]) > 0.01
    other = [2, 3, 4]  # uncensored rows are never touched
    np.testing.assert_array_equal(zdraw[:, :, other], np.broadcast_to(z[other], zdraw[:, :, other].shape))
    mc = predict_mu_draws(post_c, d_c, cfg, XS, np.random.default_rng(1))
    mu_ = predict_mu_draws(post_u, d_u, cfg, XS, np.random.default_rng(2))
    lc, hc = np.quantile(mc, [0.05, 0.95], axis=0)
    lu, hu = np.quantile(mu_, [0.05, 0.95], axis=0)
    assert np.all((lc <= hu) & (lu <= hc)), (lc, hc, lu, hu)
    assert np.all((lc <= mu_true) & (mu_true <= hc))


# an informative design for the order tests: 4 levels (hbar = 1 ... 1/8), small noise, strong h-dependence
TRUTH_ORDER = dict(
    TRUTH, m_s=np.log(0.003**2), sigma_delta=np.array([0.05]), c0=np.array([2.0]), c1=np.array([0.5])
)
SCALES_ORDER = PriorScales(S_mu=1.0, S_c=2.0, S_delta=0.2, S_noise=0.01)


def design_order():
    levels = [1.0, 0.5, 0.25, 0.125]
    X = np.concatenate([np.linspace(0.05, 0.95, 5) for _ in levels])[:, None]
    H = np.concatenate([np.full(5, h) for h in levels])[:, None]
    return X, H


def fit_order(seed, varying_truth, n_warmup, n_samples):
    rng = np.random.default_rng(7)
    X, H = design_order()
    # log p(x) = log 1.5 + 0.9 sin(2 pi x) when the truth has a varying order
    P = np.exp(np.log(1.5) + 0.9 * np.sin(2 * np.pi * X)) if varying_truth else None
    z, _ = simulate(rng, TRUTH_ORDER, X, H, XS, P=P)
    data, zp = pad(X, H, z)
    return inference.fit(
        jax.random.key(seed),
        data,
        zp,
        zp,
        ModelConfig(),
        SCALES_ORDER,
        1,
        n_warmup,
        n_samples,
        varying_order=True,
    )


def test_varying_order_shrinks_sigma_pi_when_the_order_is_constant():
    post = fit_order(4, False, 250, 250)
    assert post.params.P.shape[:2] == (4, 250)
    sig = np.exp(np.asarray(post.theta["log_sigma_pi"])).reshape(-1)
    # PC prior: P(sigma_pi > 0.3) = 0.05 exactly. Data from a constant order push mass below it.
    assert np.mean(sig > 0.3) < 0.05, np.mean(sig > 0.3)
    assert np.quantile(sig, 0.95) < 0.3


def test_varying_order_is_ignored_for_lb():
    rng = np.random.default_rng(5)
    X, H = design_1d(5, 4, 3, 2)
    z, _ = simulate(rng, TRUTH, X, H, XS)
    data, zp = pad(X, H, z)
    sampler = inference._Sampler(data, zp, zp, ModelConfig(h_kernel="lb"), SCALES, 1, True)
    assert not sampler.varying and "logit_gamma" in dict(sampler.layout.fields)
    st = sampler.draw_prior(jax.random.key(0))
    assert "pi" not in st and bool(jnp.isfinite(sampler.log_density(st)))
    P = np.asarray(sampler.constrain(st).P)
    np.testing.assert_allclose(P, np.broadcast_to(P[:1], P.shape), rtol=1e-12)  # one shared order


# ----------------------------------------------------------------------------------------------
# slow statistical tests
# ----------------------------------------------------------------------------------------------


def np_pc_sample(rng, sigma0, ell0, n_ell, n=None, alpha=0.05):
    """(log sigma, log ell) from the PC prior: sigma ~ Exp(l2); rho = ell^{-1/2} ~ Exp(l1).

    One draw (log sigma scalar, log ell of shape (n_ell,)), or n draws (shapes (n,), (n, n_ell))."""
    l1, l2 = -np.log(alpha) * np.sqrt(ell0), -np.log(alpha) / sigma0
    shape_s = () if n is None else (n,)
    log_sigma = np.log(rng.exponential(1.0 / l2, size=shape_s))
    log_ell = -2.0 * np.log(rng.exponential(1.0 / l1, size=shape_s + (n_ell,)))
    return log_sigma, log_ell


def thinned_ks(draws, ref):
    """KS p-value of pooled draws vs reference draws; draws thinned to the bulk ESS."""
    x = np.asarray(draws).reshape(draws.shape[0], draws.shape[1])
    n_eff = int(min(dg.bulk_ess(x), x.size))
    pooled = x.reshape(-1)
    step = max(1, pooled.size // max(n_eff, 20))
    # interleave chains so that thinning picks different iterations of every chain
    sub = x[:, ::step].reshape(-1)
    return stats.ks_2samp(sub, ref).pvalue, sub.size


@pytest.mark.slow
def test_prior_only_sampling_reproduces_prior_marginals():
    rng = np.random.default_rng(11)
    X, H = design_1d(5, 4, 3, 2)
    z, _ = simulate(rng, TRUTH, X, H, XS)
    data, zp = pad(X, H, z)
    sc = SCALES
    post = inference._fit_impl(
        jax.random.key(5),
        data,
        zp,
        zp,
        ModelConfig(),
        sc,
        1,
        500,
        4000,
        4,
        varying_order=True,
        likelihood_weight=0.0,
    )
    R = 40000
    ref = {}
    ref["m_s"] = rng.normal(2 * np.log(sc.S_noise), np.log(10.0), R)
    ref["b_s"] = rng.normal(0.0, sc.b_s_sd, R)
    ref["c0"], ref["c1"] = rng.normal(0.0, sc.S_c, R), rng.normal(0.0, sc.S_c, R)
    ref["log_p0"] = rng.normal(sc.log_p_mean, sc.log_p_sd, R)
    ref["log_ell_h"] = rng.normal(np.log(0.5), 1.0, R)
    s, e = np_pc_sample(rng, sc.S_mu, sc.ell0, 1, R)
    ref["log_sigma_mu"], ref["log_ell_mu"] = s, e
    s, e = np_pc_sample(rng, sc.S_delta, sc.ell0, 1, R)
    ref["log_sigma_delta"], ref["log_ell_x"] = s, e
    s, e = np_pc_sample(rng, 1.0, sc.ell0, 2, R)
    ref["log_sigma_z"], ref["log_ell_z"] = s, e
    s, e = np_pc_sample(rng, 0.3, 0.1, 1, R)
    ref["log_sigma_pi"], ref["log_ell_pi"] = s, e
    pvals = {}
    for name, r in ref.items():
        v = np.asarray(post.theta[name])
        if v.ndim == 3:
            v = v[..., 0]
        r = r if r.ndim == 1 else r[:, 0]
        pvals[name], _ = thinned_ks(v, r)
    print({k: round(float(p), 4) for k, p in pvals.items()})
    assert min(pvals.values()) > 0.001, pvals


def sbc_prior_draw(rng, sc, X, H):
    """One draw of all parameters from the prior (spec 2.5), constant order, k = d = 1, Gaussian beta."""
    T = {}
    ls, le = np_pc_sample(rng, sc.S_mu, sc.ell0, 1)
    T["sigma_mu"], T["ell_mu"] = np.exp(ls), np.exp(le)
    T["beta0"] = rng.normal(0.0, 1.0)
    T["c0"], T["c1"] = rng.normal(0, sc.S_c, 1), rng.normal(0, sc.S_c, 1)
    T["p0"] = np.exp(rng.normal(sc.log_p_mean, sc.log_p_sd, 1))
    ls, le = np_pc_sample(rng, sc.S_delta, sc.ell0, 1)
    T["sigma_delta"], T["ell_x"] = np.array([np.exp(ls)]), np.exp(le)[None, :]
    T["ell_h"] = np.exp(rng.normal(np.log(0.5), 1.0, 1))
    T["m_s"] = rng.normal(2 * np.log(sc.S_noise), np.log(10.0))
    T["b_s"] = rng.normal(0.0, sc.b_s_sd, 1)
    # latent log-variance field on the sites (u, hbar)
    sites = np.unique(np.column_stack([X, H]), axis=0)
    lsz, lez = np_pc_sample(rng, 1.0, sc.ell0, 2)
    zeta_site = draw_gp(rng, np.exp(2 * lsz) * np_ard(sites, sites, np.exp(lez), 2.5))
    key = {tuple(s): i for i, s in enumerate(sites)}
    zeta_row = np.array([zeta_site[key[tuple(r)]] for r in np.column_stack([X, H])])
    return T, zeta_row


@pytest.mark.slow
def test_simulation_based_calibration_light():
    n_rep, n_draw = 30, 100
    sc = PriorScales(S_mu=1.0, S_c=1.0, S_delta=0.5, S_noise=0.1)
    cfg = ModelConfig(beta_prior=((0.0,), (1.0,)))
    X, H = design_1d(6, 5, 4, 2)
    xstar = np.array([[0.5]])
    ranks_p, ranks_mu, fails = [], [], 0
    for r in range(n_rep):
        rng = np.random.default_rng(1000 + r)
        T, zeta_row = sbc_prior_draw(rng, sc, X, H)
        z, mu_true = simulate(rng, T, X, H, xstar, zeta_row=zeta_row)
        data, zp = pad(X, H, z)
        try:
            post = inference.fit(jax.random.key(r), data, zp, zp, cfg, sc, 1, n_warmup=200, n_samples=250)
        except RuntimeError:
            fails += 1
            continue
        lp = np.asarray(post.theta["log_p0"]).reshape(-1)
        mu = predict_mu_draws(post, data, cfg, xstar, rng, thin=1)[:, 0]
        idx = np.linspace(0, lp.size - 1, n_draw).astype(int)  # evenly thinned pooled draws
        ranks_p.append(int(np.sum(lp[idx] < np.log(T["p0"][0]))))
        ranks_mu.append(int(np.sum(mu[idx] < mu_true[0])))
    n = len(ranks_p)
    out = {}
    for name, rk in (("log_p0", ranks_p), ("mu(x*)", ranks_mu)):
        counts = np.histogram(rk, bins=np.linspace(0, n_draw + 1, 6))[0]
        out[name] = (stats.chisquare(counts).pvalue, counts.tolist())
    print(f"SBC: {n} fitted, {fails} failed; (p, counts) = {out}")
    assert fails == 0
    assert all(p > 0.01 for p, _ in out.values()), out


@pytest.mark.slow
def test_varying_order_is_detected_when_the_truth_varies():
    const = fit_order(4, False, 500, 500)
    vary = fit_order(5, True, 500, 500)
    p_const = np.mean(np.exp(np.asarray(const.theta["log_sigma_pi"])) > 0.3)
    p_vary = np.mean(np.exp(np.asarray(vary.theta["log_sigma_pi"])) > 0.3)
    print(f"P(sigma_pi > 0.3): constant truth {p_const:.3f}, varying truth {p_vary:.3f}")
    assert p_const < 0.05 < 0.2 < p_vary


# ----------------------------------------------------------------------------------------------
# S1: several outputs per run with within-run correlated noise (cfg.within_run)
# ----------------------------------------------------------------------------------------------

VS = np.array([0.2, 0.5, 0.8])  # the output coordinate v of the 3 outputs of every run


def simulate_s1(rng, ell_v, sd=0.15, n_runs=(6, 5, 4)):
    """Runs at 3 levels; each run returns 3 outputs at v = 0.2, 0.5, 0.8 with noise N(0, sd^2 R), R
    Matern-3/2 in |v - v'| / ell_v. X = (u, v), one resolution component."""
    u = np.concatenate([np.linspace(0.1, 0.9, m) for m in n_runs])
    h = np.concatenate([np.full(m, v) for m, v in zip(n_runs, (1.0, 0.5, 0.25))])
    X = np.column_stack([np.repeat(u, 3), np.tile(VS, len(u))])
    H = np.repeat(h, 3)[:, None]
    run = np.repeat(np.arange(len(u)), 3)
    T = dict(
        TRUTH,
        ell_mu=np.array([0.4, 0.5]),
        ell_x=np.array([[0.5, 0.5]]),
        m_s=np.log(sd**2),
    )
    Xs = np.array([[0.3, 0.5], [0.5, 0.2], [0.5, 0.8], [0.7, 0.5]])
    z, mu_true = simulate(rng, T, X, H, Xs, add_noise=False)
    return X, H, z, run, mu_true, Xs, T


def fit_s1(ell_v, seed, n_warmup=300, n_samples=300):
    rng = np.random.default_rng(31)
    X, H, z0, run, mu_true, Xs, T = simulate_s1(rng, ell_v)
    # the within-run correlated noise (simulate() was called without noise)
    rng2 = np.random.default_rng(32)
    z = z0.copy()
    R = np_matern(np.abs(VS[:, None] - VS[None, :]) / ell_v, 1.5)
    Lr = np.linalg.cholesky(R + 1e-10 * np.eye(3))
    sd = np.exp(0.5 * T["m_s"])
    for r in range(run.max() + 1):
        idx = np.flatnonzero(run == r)
        z[idx] += sd * (Lr @ rng2.standard_normal(3))
    data, zp = pad(X, H, z, run=run)
    cfg = ModelConfig(within_run=True, v_index=(1,))
    sc = PriorScales(S_mu=1.0, S_c=1.0, S_delta=0.5, S_noise=0.15)
    post = inference.fit(jax.random.key(seed), data, zp, zp, cfg, sc, 1, n_warmup, n_samples)
    return data, cfg, post, mu_true, Xs


@pytest.mark.slow
def test_within_run_correlation_is_learned_and_mu_recovered():
    out = {}
    for name, ell_v in (("strong", 5.0), ("none", 0.05)):
        data, cfg, post, mu_true, Xs = fit_s1(ell_v, 8)
        mu = predict_mu_draws(post, data, cfg, Xs, np.random.default_rng(0))
        lo, hi = np.quantile(mu, [0.005, 0.995], axis=0)
        assert np.all((lo <= mu_true) & (mu_true <= hi)), (name, lo, mu_true, hi)
        out[name] = np.exp(np.asarray(post.theta["log_ell_v"])).reshape(-1)
    # prior: ell_v = rho^-2, rho ~ Exp(l1): median 1/ (ln2 / l1)^2
    l1 = -np.log(0.05) * np.sqrt(0.1)
    prior_median = (np.log(2.0) / l1) ** -2
    print(
        f"ell_v medians: strong {np.median(out['strong']):.2f}, none {np.median(out['none']):.2f}, "
        f"prior {prior_median:.2f}"
    )
    assert np.median(out["none"]) < 0.5 * prior_median  # weak correlation: pulled well below the prior
    assert (
        np.mean(out["strong"] > 3.0) > np.mean(out["none"] > 3.0) + 0.3
    )  # strong: mass moves to large ell_v


@pytest.mark.slow
def test_noise_hyperparameters_converge_at_n30_with_the_non_centred_move():
    rng = np.random.default_rng(0)
    m = 10
    X = np.concatenate([rng.uniform(0.02, 0.98, m) for _ in range(3)])[:, None]
    H = np.concatenate([np.full(m, h) for h in (1.0, 0.5, 0.25)])[:, None]
    z, _ = simulate(rng, TRUTH, X, H, XS)
    data, zp = pad(X, H, z)
    post = inference.fit(jax.random.key(3), data, zp, zp, ModelConfig(), SCALES, 1, 1000, 1000)
    for name in ("log_sigma_z", "log_p0[0]"):
        rh, bulk, tail = post.diagnostics[name]
        print(name, round(rh, 3), round(bulk), round(tail))
        assert rh < 1.01 and bulk > 400 and tail > 400, (name, rh, bulk, tail)
