"""Error shapes b(hbar): "power", "saturating", "two_term" (spec.md, "Error shape (optional)")."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from test_inference import SCALES, TRUTH, XS, design_1d, pad, simulate

import gcbml  # noqa: F401  (float64)
from gcbml import inference, kernels, model
from gcbml.kernels import DeltaParams
from gcbml.mcmc import diagnostics as dg
from gcbml.model import ModelConfig, ModelParams

# Reference values of the code BEFORE the error-shape option existed (preasym branch, commit 560d6d4),
# on the dataset of world(): log_marginal and predict_mu of the power law.
REF_LM = 4.610889645151968
REF_MU = [
    0.4646699339458299,
    0.6200191047969442,
    0.5150962537241225,
    0.10020322474353197,
    -0.2378009628623284,
]
REF_VAR = [
    0.01018584762376227,
    0.011213384486022488,
    0.005079409607901804,
    0.011642213217471876,
    0.010360277770762996,
]


def world(aux=None):
    rng = np.random.default_rng(7)
    X, H = design_1d(5, 4, 3, 2)
    z, _ = simulate(rng, TRUTH, X, H, XS)
    data, zp = pad(X, H, z)
    n_pad = data.X.shape[0]
    kw = {} if aux is None else dict(aux=jnp.asarray(aux, dtype=float))
    params = ModelParams(
        sigma_mu=jnp.asarray(0.8),
        ell_mu=jnp.asarray([0.4]),
        c0=jnp.asarray([0.6]),
        c1=jnp.asarray([-0.3]),
        P=jnp.full((n_pad, 1), 1.3),
        delta=DeltaParams(jnp.asarray([0.3]), jnp.asarray([[0.5]]), jnp.asarray([0.5]), jnp.asarray([0.5])),
        noise_var=jnp.full((n_pad,), 0.05**2),
        ell_v=jnp.asarray(1.0),
        **kw,
    )
    return data, jnp.asarray(zp), params


def evaluate(cfg, aux=None):
    data, z, params = world(aux)
    lm = float(model.log_marginal(params, data, z, cfg))
    m, v = model.predict_mu(params, data, z, cfg, jnp.asarray(XS))
    return lm, np.asarray(m), np.asarray(v)


def test_a_power_reproduces_the_existing_code():
    lm, m, v = evaluate(ModelConfig(shape="power"), aux=np.zeros(2))
    assert abs(lm - REF_LM) < 1e-12 * max(1.0, abs(REF_LM))
    np.testing.assert_allclose(m, REF_MU, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v, REF_VAR, rtol=0, atol=1e-12)
    # the default config and the default aux are the power law too
    lm0, m0, _ = evaluate(ModelConfig())
    assert lm0 == lm
    np.testing.assert_array_equal(m0, m)


def test_b_saturating_with_huge_h_s_is_power():
    lm0, m0, v0 = evaluate(ModelConfig(shape="power"))
    lm, m, v = evaluate(ModelConfig(shape="saturating"), aux=[30.0, 0.0])
    assert abs(lm - lm0) < 1e-9
    np.testing.assert_allclose(m, m0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(v, v0, rtol=0, atol=1e-9)


def test_c_two_term_with_zero_weight_is_power():
    lm0, m0, v0 = evaluate(ModelConfig(shape="power"))
    lm, m, v = evaluate(ModelConfig(shape="two_term"), aux=[0.0, 0.7])
    assert abs(lm - lm0) < 1e-12
    np.testing.assert_allclose(m, m0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v, v0, rtol=0, atol=1e-12)


SHAPES = (("power", [0.0, 0.0]), ("saturating", [-0.3, 0.0]), ("two_term", [0.8, 0.4]))


@pytest.mark.parametrize("shape,aux", SHAPES)
def test_d_zero_at_zero_and_finite_gradients(shape, aux):
    aux = jnp.asarray(aux)
    h = jnp.asarray([0.0, 0.5, 1.0])
    b = kernels.err_shape(h, 1.3, shape, aux)
    assert float(b[0]) == 0.0
    assert np.all(np.isfinite(np.asarray(b)))

    def f(h, p, aux):
        return jnp.sum(kernels.err_shape(h, p, shape, aux))

    g = jax.grad(f, argnums=(0, 1, 2))(h, 1.3, aux)
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))
    # the gradient with respect to h at h = 0 alone (the masked branch must not leak a nan)
    g0 = jax.grad(lambda x: kernels.err_shape(x, 1.3, shape, aux))(jnp.asarray(0.0))
    assert np.isfinite(float(g0))


def test_e_saturating_limits():
    p, log_hs = 1.4, 0.0  # h_s = 1
    aux = jnp.asarray([log_hs, 0.0])
    small = jnp.asarray([1e-3, 3e-3])
    np.testing.assert_allclose(
        kernels.err_shape(small, p, "saturating", aux), np.asarray(small) ** p, rtol=1e-9
    )
    big = jnp.asarray([1e3, 1e4])
    np.testing.assert_allclose(kernels.err_shape(big, p, "saturating", aux), np.exp(p * log_hs), rtol=1e-9)
    # another h_s and sharpness
    aux2 = jnp.asarray([np.log(0.4), 0.0])
    big2 = jnp.asarray([1e3])
    np.testing.assert_allclose(kernels.err_shape(big2, p, "saturating", aux2, sat_m=2.0), 0.4**p, rtol=1e-6)


def test_two_term_formula():
    w, a1 = 0.8, 0.4
    u = 1.0 / (1.0 + np.exp(-a1))
    h = np.asarray([0.25, 0.5, 1.0])
    p = 1.3
    expect = h**p + w * h ** (p * u)
    np.testing.assert_allclose(kernels.err_shape(jnp.asarray(h), p, "two_term", jnp.asarray([w, a1])), expect)


def test_unknown_shape_and_lb_are_rejected():
    with pytest.raises(ValueError):
        kernels.err_shape(jnp.asarray([0.5]), 1.0, "nope", jnp.zeros(2))
    data, z, params = world()
    with pytest.raises(ValueError):
        model.log_marginal(params, data, z, ModelConfig(h_kernel="lb", shape="saturating"))


def _small_data():
    rng = np.random.default_rng(3)
    X, H = design_1d(5, 4, 3, 2)
    z, _ = simulate(rng, TRUTH, X, H, XS)
    return pad(X, H, z)


def _hs_bounds(data):
    H = np.asarray(data.H)[np.asarray(data.mask)]
    pos = H[H > 0]
    return np.log(pos.min() / 2.0), np.log(8.0 * pos.max())


@pytest.mark.parametrize("shape", ["saturating", "two_term"])
def test_f_short_fit_runs_and_aux_is_finite_and_in_support(shape):
    data, zp = _small_data()
    cfg = ModelConfig(shape=shape)
    post = inference.fit(
        jax.random.key(2), data, zp, zp, cfg, SCALES, n_controls=1, n_warmup=30, n_samples=30,
        n_chains=2, max_extensions=0,
    )  # fmt: skip
    n_aux = 1 if shape == "saturating" else 2
    assert post.theta["aux"].shape == (2, 30, n_aux)
    assert np.all(np.isfinite(post.theta["aux"]))
    assert post.params.aux.shape == (2, 30, 2)
    assert np.all(np.isfinite(np.asarray(post.params.aux)))
    assert any(name.startswith("aux") for name in post.diagnostics)
    if shape == "saturating":
        lo, hi = _hs_bounds(data)
        a = post.theta["aux"][..., 0]
        assert np.all((a >= lo) & (a <= hi)), (lo, a.min(), a.max(), hi)
        np.testing.assert_array_equal(np.asarray(post.params.aux)[..., 1], 0.0)
    params, z = inference.flatten(post)
    m, v = jax.vmap(lambda p, zz: model.predict_mu(p, data, zz, cfg, jnp.asarray(XS)))(params, z)
    assert np.all(np.isfinite(np.asarray(m))) and np.all(np.isfinite(np.asarray(v)))


def _ks_thinned(draws, cdf):
    x = np.asarray(draws).reshape(draws.shape[0], draws.shape[1])
    n_eff = int(min(dg.bulk_ess(x), x.size))
    pooled = x.reshape(-1)
    step = max(1, pooled.size // max(n_eff, 20))
    return stats.kstest(x[:, ::step].reshape(-1), cdf).pvalue


def _prior_chain(shape):
    data, zp = _small_data()
    return data, inference._fit_impl(
        jax.random.key(5), data, zp, zp, ModelConfig(shape=shape), SCALES, 1, 200, 1500, 4,
        likelihood_weight=0.0,
    )  # fmt: skip


def test_g_prior_chain_saturating_log_hs_is_uniform():
    data, post = _prior_chain("saturating")
    lo, hi = _hs_bounds(data)
    a = post.theta["aux"][..., 0]
    assert np.all((a >= lo) & (a <= hi))
    p = _ks_thinned(a, stats.uniform(loc=lo, scale=hi - lo).cdf)
    assert p > 0.01, p


def test_g_prior_chain_two_term_aux_is_standard_normal():
    _, post = _prior_chain("two_term")
    for i in range(2):
        p = _ks_thinned(post.theta["aux"][..., i], stats.norm.cdf)
        assert p > 0.01, (i, p)


def test_h_lo_factor_default_reproduces_current_prior_bounds():
    data, zp = _small_data()
    assert ModelConfig().sat_lo_factor == 0.5
    sampler = inference._Sampler(data, zp, zp, ModelConfig(shape="saturating"), SCALES, 1, False)
    np.testing.assert_array_equal(np.asarray(sampler.aux.hs_bounds), np.array(_hs_bounds(data)))


def test_g_prior_chain_saturating_log_hs_uniform_with_lo_factor_one():
    data, zp = _small_data()
    post = inference._fit_impl(
        jax.random.key(6), data, zp, zp, ModelConfig(shape="saturating", sat_lo_factor=1.0), SCALES, 1,
        200, 1500, 4, likelihood_weight=0.0,
    )  # fmt: skip
    H = np.asarray(data.H)[np.asarray(data.mask)]
    lo, hi = np.log(H[H > 0].min()), np.log(8.0 * H.max())
    a = post.theta["aux"][..., 0]
    assert np.all((a >= lo) & (a <= hi))
    assert _ks_thinned(a, stats.uniform(loc=lo, scale=hi - lo).cdf) > 0.01
