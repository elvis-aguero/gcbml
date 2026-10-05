"""Plug-in GP: the universal-kriging mean and variance against an explicit (bordered) linear system."""

import math

import numpy as np

from benchmarks.gp import PlugInGP, matern52_ard


def brute_force(X, y, Xs, sf, ell, noise_var):
    """Universal kriging with a constant GLS mean from the bordered system, solved with np.linalg.solve."""
    n = len(X)
    K = sf**2 * matern52_ard(X, X, ell) + np.diag(noise_var)
    A = np.block([[K, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    mean, var = [], []
    for xs in Xs:
        k = sf**2 * matern52_ard(X, xs[None, :], ell)[:, 0]
        sol = np.linalg.solve(A, np.append(k, 1.0))
        w, mu = sol[:n], sol[n]
        mean.append(w @ y)
        var.append(sf**2 - w @ k - mu)
    return np.array(mean), np.sqrt(np.array(var))


def test_prediction_matches_the_bordered_linear_system():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    sd = 0.01 * np.ones(12)
    gp = PlugInGP.fit(X, y, known_sd=sd, fit_nugget=True)
    Xs = rng.uniform(size=(7, 2))
    ys = gp.ys
    m_ref, s_ref = brute_force(X, (y - gp.ym) / ys, Xs, gp.sf, gp.ell, gp.known_var + gp.nugget**2)
    m, s = gp.predict(Xs)
    assert np.allclose(m, gp.ym + ys * m_ref, rtol=1e-8, atol=1e-10)
    assert np.allclose(s, ys * s_ref, rtol=1e-6, atol=1e-10)


def test_interpolates_a_smooth_function_and_variance_vanishes_at_data():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(30, 2))
    f = lambda x: np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1])  # noqa: E731
    gp = PlugInGP.fit(X, f(X), fit_nugget=False)
    Xs = rng.uniform(size=(50, 2))
    m, s = gp.predict(Xs)
    assert np.max(np.abs(m - f(Xs))) < 0.05
    m0, s0 = gp.predict(X)
    assert np.max(np.abs(m0 - f(X))) < 1e-3 and np.max(s0) < 1e-2 * gp.ys + 1e-3
    # honest: the error is within a few predictive sd at most test points
    assert np.mean(np.abs(m - f(Xs)) <= 3 * s + 1e-6) > 0.8


def test_known_noise_inflates_the_predictive_sd():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(15, 1))
    y = np.sin(4 * X[:, 0])
    a = PlugInGP.fit(X, y, known_sd=np.full(15, 1e-6), fit_nugget=False)
    b = PlugInGP.fit(X, y, known_sd=np.full(15, 0.3), fit_nugget=False)
    Xs = np.linspace(0, 1, 9)[:, None]
    assert np.all(b.predict(Xs)[1] > a.predict(Xs)[1])
    assert math.isfinite(b.loglik)
