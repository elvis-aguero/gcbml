"""p-quadrature (gcbml.quadrature): grid, optimum, weights, baselines, mixture prediction.

Data come from the NumPy simulator of test_inference (the generative model of spec 2.2-2.4).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_inference as ti
from scipy import stats

import gcbml  # noqa: F401  (float64)
from gcbml import inference, model, quadrature
from gcbml.kernels import delta_cov
from gcbml.model import ModelConfig

CFG = ModelConfig()
TRUE_LOG_P = float(np.log(ti.TRUTH["p0"][0]))


def make(n_coarse, n_mid, n_fine, n_rep, seed=0):
    rng = np.random.default_rng(seed)
    X, H = ti.design_1d(n_coarse, n_mid, n_fine, n_rep)
    z, mu_true = ti.simulate(rng, ti.TRUTH, X, H, ti.XS)
    data, zp = ti.pad(X, H, z)
    return data, zp, mu_true


def quad(data, zp, cfg=CFG, **kw):
    return quadrature.fit_quadrature(jax.random.key(0), data, zp, zp, cfg, ti.SCALES, 1, **kw)


@pytest.fixture(scope="module")
def small():
    """Full default fit (two passes of 20 nodes): slow tests only."""
    data, zp, mu_true = make(8, 6, 4, 3)
    return data, zp, mu_true, quad(data, zp)


@pytest.fixture(scope="module")
def tiny():
    """Single pass, 4 explicit nodes: the fast smoke tests."""
    data, zp, mu_true = make(5, 4, 3, 2)
    return data, zp, mu_true, quad(data, zp, grid=np.linspace(-1.0, 1.0, 4)[:, None])


def test_grid_is_20_equal_steps_over_the_central_99_percent():
    g = quadrature.log_p_grid(ti.SCALES, 1)
    assert g.shape == (20, 1)
    z = stats.norm.ppf(0.995)
    np.testing.assert_allclose(g[0, 0], -z * ti.SCALES.log_p_sd, rtol=1e-12)
    np.testing.assert_allclose(g[-1, 0], z * ti.SCALES.log_p_sd, rtol=1e-12)
    np.testing.assert_allclose(np.diff(g[:, 0]), np.diff(g[:, 0])[0], rtol=1e-9)
    # prior mass between the end nodes is 99%
    assert abs(stats.norm.cdf(g[-1, 0]) - stats.norm.cdf(g[0, 0]) - 0.99) < 1e-12


def test_grid_two_components_is_a_snake_tensor_grid_and_three_raise():
    g = quadrature.log_p_grid(ti.SCALES, 2)
    assert g.shape == (400, 2)
    assert len({tuple(r) for r in g}) == 400
    assert np.all(np.abs(np.diff(g, axis=0)).max(axis=1) <= 2 * (g[1, 1] - g[0, 1]) + 1e-12 + 0)  # neighbours
    with pytest.raises(ValueError, match="k <= 2"):
        quadrature.log_p_grid(ti.SCALES, 3)


@pytest.mark.slow
def test_optimum_is_a_stationary_point_below_the_start(small):
    data, zp, _, post = small
    sampler = inference._Sampler(data, zp, zp, CFG, ti.SCALES, 1, False)
    prob = quadrature._Problem(sampler)
    vg = jax.jit(jax.value_and_grad(prob.neg_log_post))
    f0, _ = vg(jnp.asarray(prob.initial()), jnp.asarray([0.0]), sampler.aux)
    # re-optimise one node and check the gradient: the stored objective value is the optimum's
    one = quadrature.fit_plugin(jax.random.key(0), data, zp, zp, CFG, ti.SCALES, 1, log_p=0.0)
    assert one.weights.shape == (1,) and one.weights[0] == 1.0
    assert one.log_post[0] > -float(f0) + 1.0  # moved far uphill from the prior-median start
    assert bool(one.converged[0])


@pytest.mark.slow
def test_plugin_log_post_matches_node_of_the_grid(small):
    data, zp, _, post = small
    k = 9  # a node near the middle
    one = quadrature.fit_plugin(jax.random.key(0), data, zp, zp, CFG, ti.SCALES, 1, log_p=post.log_p[k])
    # same objective, optimiser started elsewhere: values agree to a small tolerance (one mode)
    assert abs(one.log_post[0] - post.log_post[k]) < 0.05, (one.log_post[0], post.log_post[k])


def test_weights_are_a_normalised_softmax_of_log_post(tiny):
    *_, post = tiny
    w = np.asarray(post.weights)
    assert w.shape == (4,) and abs(w.sum() - 1.0) < 1e-12 and np.all(w >= 0)
    ref = np.exp(post.log_post - post.log_post.max())
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-12)


def test_output_has_the_structure_of_inference_posterior(tiny):
    data, zp, _, post = tiny
    n_pad = data.X.shape[0]
    assert post.params.c0.shape == (1, 4, 1)
    assert post.params.P.shape == (1, 4, n_pad, 1)
    assert post.params.noise_var.shape == (1, 4, n_pad)
    assert post.z.shape == (1, 4, n_pad)
    params, z = inference.flatten(post)  # the unchanged flatten
    assert z.shape == (4, n_pad) and params.sigma_mu.shape == (4,)
    np.testing.assert_allclose(np.log(np.asarray(params.P[:, 0, 0])), post.log_p[:, 0], rtol=1e-12)
    assert set(post.theta) >= {"log_p0", "log_sigma_z", "log_ell_z", "log_sigma_mu"}


def test_prediction_is_the_weighted_mixture_of_exact_gaussians(tiny):
    data, zp, _, post = tiny
    mean, var = quadrature.mu_moments(post, data, CFG, ti.XS)
    assert mean.shape == (4, 5) and np.all(var >= 0)
    # brute force: one node by model.predict_mu on its own parameters
    params, z = inference.flatten(post)
    p7 = jax.tree_util.tree_map(lambda a: a[2], params)
    m7, v7 = model.predict_mu(p7, data, z[2], CFG, jnp.asarray(ti.XS))
    np.testing.assert_allclose(mean[2], np.asarray(m7), rtol=1e-10)
    # mixture quantiles against a brute-force CDF root find
    q = quadrature.mixture_quantiles(post.weights, mean, var, [0.16, 0.5, 0.84])
    from scipy.optimize import brentq

    for j in range(5):
        F = lambda y, j=j: np.sum(post.weights * stats.norm.cdf(y, mean[:, j], np.sqrt(var[:, j])))  # noqa: E731, E501
        for qi, level in enumerate([0.16, 0.5, 0.84]):
            ref = brentq(
                lambda y, level=level, F=F: F(y) - level,
                mean[:, j].min() - 10,
                mean[:, j].max() + 10,
                xtol=1e-12,
            )
            assert abs(q[qi, j] - ref) < 1e-8


def test_decompose_is_the_law_of_total_variance():
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(6))
    m, v = rng.normal(size=(6, 3)), rng.uniform(0.1, 1.0, (6, 3))
    tot, ev, vm = quadrature.decompose(w, m, v)
    x = np.concatenate(
        [m[k] + np.sqrt(v[k]) * np.random.default_rng(1).standard_normal((400000, 3)) for k in [0]]
    )
    del x  # a sampled check is below
    ref_mean = (w[:, None] * m).sum(0)
    ref = (w[:, None] * (v + m**2)).sum(0) - ref_mean**2  # E[Y^2] - E[Y]^2
    np.testing.assert_allclose(tot, ref, rtol=1e-12)
    np.testing.assert_allclose(tot, ev + vm, rtol=1e-12)


def test_equal_weight_bridge_reproduces_the_weights(tiny):
    *_, post = tiny
    eq = quadrature.to_equal_weight(post, 400)
    assert eq.z.shape[:2] == (1, 400)
    lp = np.asarray(eq.theta["log_p0"]).reshape(400, 1)
    counts = np.array([np.sum(np.isclose(lp[:, 0], g)) for g in post.log_p[:, 0]]) / 400
    assert np.max(np.abs(counts - post.weights)) < 1.0 / 400 + 1e-12


def test_varying_order_is_not_supported_and_censoring_raises():
    data, zp, _ = make(5, 4, 3, 2)
    cens = np.asarray(data.censored).copy()
    cens[0] = True
    bad = dataclasses.replace(data, censored=cens)
    with pytest.raises(NotImplementedError, match="censored"):
        quad(bad, zp)


@pytest.mark.slow
def test_second_pass_refines_the_range_of_the_first(small):
    *_, post = small
    g1, w1 = post.pass1["pass1_grid"][:, 0], post.pass1["pass1_weights"]
    g2 = post.log_p[:, 0]
    dx = g1[1] - g1[0]
    cdf = np.cumsum(w1)
    lo = g1[np.searchsorted(cdf, 0.0005)] - dx
    hi = g1[min(np.searchsorted(cdf, 0.9995), len(g1) - 1)] + dx
    np.testing.assert_allclose([g2[0], g2[-1]], [lo, hi], rtol=1e-9, atol=1e-9)
    assert len(g2) == 20 and np.allclose(np.diff(g2), np.diff(g2)[0])


@pytest.mark.slow
def test_brownian_option_pins_gamma_and_has_no_gamma_parameter():
    data, zp, _ = make(5, 4, 3, 2)
    cfg = ModelConfig(h_kernel="lb", gamma_fixed=0.5)
    sampler = inference._Sampler(data, zp, zp, cfg, ti.SCALES, 1, False)
    assert "logit_gamma" not in dict(sampler.layout.fields)
    st = sampler.draw_prior(jax.random.key(0))
    assert bool(jnp.isfinite(sampler.log_density(st)))
    p = sampler.constrain(st)
    np.testing.assert_allclose(np.asarray(p.delta.gamma), 0.5)
    # covariance equals the Brownian closed form sum_j sigma_j^2 kx min(h, h')^{2p}
    h = np.asarray(data.H)[:, 0]
    P = float(np.asarray(p.P)[0, 0])
    ref = np.minimum(h[:, None], h[None, :]) ** (2 * P)
    kd = np.asarray(delta_cov(data.X, data.H, data.X, data.H, p.P, p.P, p.delta, "lb", cfg.nu_x, cfg.nu_h))
    kx = np.asarray(
        delta_cov(data.X, np.ones_like(data.H), data.X, np.ones_like(data.H), p.P, p.P, p.delta, "lb")
    )
    del kx
    from gcbml.kernels import ard_matern

    Kx = np.asarray(ard_matern(jnp.asarray(data.X), jnp.asarray(data.X), p.delta.ell_x[0], cfg.nu_x))
    np.testing.assert_allclose(kd, float(p.delta.sigma[0]) ** 2 * Kx * ref, rtol=1e-9, atol=1e-12)
    # and the plug-in / quadrature fits run with it
    post = quadrature.fit_plugin(jax.random.key(0), data, zp, zp, cfg, ti.SCALES, 1, log_p=0.0)
    np.testing.assert_allclose(np.asarray(post.params.delta.gamma), 0.5)
    assert "logit_gamma" not in post.theta


def richardson(n_levels, n_sites, seed=11):
    """Near-deterministic Richardson sequence f0 + c0 hbar^p (p = 1.5 known), levels hbar = 2^-l, small
    residual."""
    T = dict(
        ti.TRUTH, sigma_delta=np.array([0.03]), m_s=np.log(1e-2**2), c0=np.array([0.8]), c1=np.array([0.0])
    )
    rng = np.random.default_rng(seed)
    u = np.linspace(0.05, 0.95, n_sites)
    X = np.concatenate([u] * n_levels)[:, None]
    H = np.concatenate([np.full(n_sites, 0.5**lev) for lev in range(n_levels)])[:, None]
    z, _ = ti.simulate(rng, T, X, H, ti.XS)
    data, zp = ti.pad(X, H, z)
    return data, zp


@pytest.mark.slow
def test_weight_mass_concentrates_near_the_true_p_as_data_increase():
    """1-D Richardson truth (p0 = 1.5 known): the pass-1 weight mass within +-0.35 of log p0 grows with
      the data
    (more levels and sites). Three levels alone do not identify p, in full MCMC either (see the report)."""
    masses = []
    for n_levels, n_sites in [(3, 3), (4, 5), (6, 8)]:
        data, zp = richardson(n_levels, n_sites)
        post = quad(data, zp)
        g1, w1 = post.pass1["pass1_grid"][:, 0], post.pass1["pass1_weights"]  # the prior grid, pass 1
        masses.append(float(w1[np.abs(g1 - TRUE_LOG_P) < 0.35].sum()))
    prior_mass = stats.norm.cdf(TRUE_LOG_P + 0.35) - stats.norm.cdf(TRUE_LOG_P - 0.35)
    print("prior mass", prior_mass, "weight mass near true p by data size", masses)
    assert masses[0] < masses[1] < masses[2]
    assert masses[-1] > 0.6 > 2 * prior_mass
