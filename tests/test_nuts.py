"""NUTS-in-Gibbs (gcbml.nuts) and the derivatives it needs.

Known answers: the JAX gradient of model.log_marginal against central finite differences (an independent
numerical reference), and the posterior of the Gibbs/slice sampler (inference.fit) as the reference
distribution of the same model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_inference as ti
from scipy import stats

import gcbml  # noqa: F401  (float64)
from gcbml import inference, nuts
from gcbml.model import ModelConfig

CFG = ModelConfig()


def make(seed=0, sizes=(8, 6, 4, 3)):
    rng = np.random.default_rng(seed)
    X, H = ti.design_1d(*sizes)
    z, _ = ti.simulate(rng, ti.TRUTH, X, H, ti.XS)
    return ti.pad(X, H, z)


def test_theta_gradient_matches_central_finite_differences():
    data, zp = make()
    sampler = inference._Sampler(data, zp, zp, CFG, ti.SCALES, 1, False)
    st = sampler.draw_prior(jax.random.key(3))
    pi = sampler._no_pi()

    @jax.jit
    def f(v):
        return sampler._theta_logdensity(v, st["hz"], st["zeta"], pi, st["z"])

    v0 = st["theta"]
    g = np.asarray(jax.grad(f)(v0))
    assert np.all(np.isfinite(g))
    h = 1e-6
    eye = jnp.eye(v0.shape[0])
    fd = np.asarray(jax.vmap(lambda e: (f(v0 + h * e) - f(v0 - h * e)) / (2 * h))(eye))
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-5)


def test_ard_matern_gradient_is_finite_on_the_diagonal():
    from gcbml.kernels import ard_matern

    X = jnp.asarray(np.random.default_rng(0).random((6, 2)))
    g = jax.grad(lambda ell: jnp.sum(ard_matern(X, X, ell, 2.5)))(jnp.array([0.3, 0.5]))
    assert np.all(np.isfinite(np.asarray(g)))


@pytest.fixture(scope="module")
def both():
    data, zp = make(1)
    gibbs = inference.fit(jax.random.key(1), data, zp, zp, CFG, ti.SCALES, 1, 400, 400, 4)
    nu = nuts.fit_nuts(jax.random.key(2), data, zp, zp, CFG, ti.SCALES, 1, 250, 400, 4)
    return data, zp, gibbs, nu


@pytest.mark.slow
def test_nuts_posterior_has_the_structure_of_inference_posterior(both):
    data, _, gibbs, nu = both
    n_pad = data.X.shape[0]
    assert nu.params.c0.shape == gibbs.params.c0.shape == (4, 400, 1)
    assert nu.params.P.shape == gibbs.params.P.shape
    assert nu.z.shape == (4, 400, n_pad)
    assert set(nu.theta) == set(gibbs.theta)
    p, z = inference.flatten(nu)
    assert z.shape == (1600, n_pad)
    assert nu.info["divergences"].sum() <= 0.1 * 1600  # 62 of 1600 measured (funnel geometry)
    assert np.all(nu.info["mean_accept"] > 0.6)


@pytest.mark.slow
def test_nuts_agrees_with_the_slice_sampler_on_the_global_settings(both):
    _, _, gibbs, nu = both
    for name in ("log_p0", "log_sigma_mu", "c0", "m_s"):
        a = np.asarray(gibbs.theta[name]).reshape(-1)
        b = np.asarray(nu.theta[name]).reshape(-1)
        ks = stats.ks_2samp(a, b).statistic
        assert ks < 0.12, (name, ks)
        assert abs(a.mean() - b.mean()) < 0.25 * a.std(), (name, a.mean(), b.mean(), a.std())
    assert max(v[0] for v in nu.diagnostics.values()) < 1.1
