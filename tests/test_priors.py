import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

import gcbml  # noqa: F401
from gcbml import priors


def pc_numpy(u, v, sigma0, ell0, a_s, a_l):
    """Fuglstad et al. 1503.00256 Thm 2.6 with d = 1, written in (sigma, ell), plus exp Jacobians."""
    sigma, ell = math.exp(u), math.exp(v)
    l1 = -math.log(a_l) * ell0**0.5
    l2 = -math.log(a_s) / sigma0
    dens = 0.5 * l1 * l2 * ell**-1.5 * math.exp(-l1 * ell**-0.5 - l2 * sigma)
    return math.log(dens) + u + v


def test_pc_matches_hand_formula():
    for u, v in [(-1.0, -2.0), (0.3, 0.5), (1.2, -0.7)]:
        got = float(priors.pc_matern_logpdf(u, v, 1.5, 0.2, 0.05, 0.1))
        assert abs(got - pc_numpy(u, v, 1.5, 0.2, 0.05, 0.1)) < 1e-12


def _grid_density(sigma0, ell0, a_s, a_l):
    du, dv = 0.02, 0.05
    u = np.arange(-25.0, 8.0 + du / 2, du)
    v = np.arange(-30.0, 120.0 + dv / 2, dv)
    f = jax.vmap(
        jax.vmap(lambda a, b: priors.pc_matern_logpdf(a, b, sigma0, ell0, a_s, a_l), (None, 0)), (0, None)
    )
    return u, v, np.exp(np.asarray(f(jnp.asarray(u), jnp.asarray(v))))


def test_pc_density_integrates_to_one_and_calibrates():
    sigma0, ell0 = math.exp(0.5), math.exp(-2.0)
    a_s, a_l = 0.05, 0.1
    u, v, p = _grid_density(sigma0, ell0, a_s, a_l)
    total = np.trapezoid(np.trapezoid(p, v, axis=1), u)
    assert abs(total - 1.0) < 1e-4
    iu, iv = np.argmin(abs(u - 0.5)), np.argmin(abs(v + 2.0))
    p_sigma = np.trapezoid(np.trapezoid(p[iu:], v, axis=1), u[iu:])
    p_ell = np.trapezoid(np.trapezoid(p[:, : iv + 1], v[: iv + 1], axis=1), u)
    assert abs(p_sigma - a_s) < 1e-4
    assert abs(p_ell - a_l) < 1e-4


def test_pc_ard_sums_d_length_scale_factors_and_one_sigma_factor():
    u, vs = 0.4, np.array([-1.0, 0.2, -2.5, 0.9])
    sigma0, ell0 = 2.0, 0.1
    # one sigma factor: log l2 + u - l2 sigma ; one ell factor each: log(0.5 l1) - 0.5 v - l1 e^{-v/2}
    l1, l2 = -math.log(0.05) * ell0**0.5, -math.log(0.05) / sigma0
    ref = math.log(l2) + u - l2 * math.exp(u)
    ref += sum(math.log(0.5 * l1) - 0.5 * v - l1 * math.exp(-0.5 * v) for v in vs)
    got = float(priors.pc_matern_logpdf(u, jnp.asarray(vs), sigma0, ell0))
    assert abs(got - ref) < 1e-12
    # consistent with the d = 1 density: ard = sum_i pc(u, v_i) - (d - 1) sigma factor
    sig = math.log(l2) + u - l2 * math.exp(u)
    alt = sum(float(priors.pc_matern_logpdf(u, v, sigma0, ell0)) for v in vs) - (len(vs) - 1) * sig
    assert abs(got - alt) < 1e-12


def test_lognormal_matches_scipy_with_change_of_variables():
    log_x = np.array([-1.0, 0.3, 2.0])
    mu, sd = 0.4, 0.8
    got = float(priors.lognormal_logpdf(jnp.asarray(log_x), mu, sd))
    # density of x is scipy lognorm; density of log x = lognorm.pdf(x) * x
    x = np.exp(log_x)
    ref = np.sum(stats.lognorm.logpdf(x, s=sd, scale=math.exp(mu)) + log_x)
    assert abs(got - ref) < 1e-12


def test_normal_matches_scipy_and_sums_trailing_dims():
    x = np.random.default_rng(0).normal(size=(3, 4))
    got = priors.normal_logpdf(jnp.asarray(x), 0.5, 1.7)
    assert got.shape == ()
    assert abs(float(got) - stats.norm.logpdf(x, 0.5, 1.7).sum()) < 1e-12


def test_logit_uniform_matches_scipy_with_change_of_variables():
    t = np.array([-3.0, -0.2, 0.0, 1.5, 4.0])
    got = float(priors.logit_uniform_logpdf(jnp.asarray(t)))
    x = 1 / (1 + np.exp(-t))
    # x ~ U(0,1): density of t = uniform.pdf(x) * dx/dt, with dx/dt = x (1 - x)
    ref = np.sum(stats.uniform.logpdf(x) + np.log(x * (1 - x)))
    assert abs(got - ref) < 1e-12
    # integrates to 1 (logistic distribution density)
    grid = np.linspace(-40, 40, 200001)
    d = np.exp(np.asarray(jax.vmap(priors.logit_uniform_logpdf)(jnp.asarray(grid))))
    assert abs(np.trapezoid(d, grid) - 1.0) < 1e-8


def test_priors_jit():
    f = jax.jit(lambda a, b: priors.pc_matern_logpdf(a, b, 1.0, 0.1))
    assert np.isfinite(float(f(0.1, jnp.array([-1.0, -2.0]))))
