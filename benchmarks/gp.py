"""Plug-in Gaussian process for the baselines: ARD Matern 5/2, constant mean, ML-II hyperparameters.

Model: y(x) = beta + sf * g(x) + noise, g a unit-variance Matern 5/2 field with one length scale per input
(unit-cube inputs), noise variance = known_i^2 (given per point, may be zero) + nugget^2.
Hyperparameters (log sf, log ell_1..d, log nugget) maximise the likelihood with beta profiled out (GLS);
y is standardised first so one set of bounds serves every problem. Multi-start L-BFGS-B, fixed seed.

Prediction is the universal-kriging (plug-in) latent mean and sd of f(x*), the noise-free function:

    var f(x*) = sf^2 - k*' K^-1 k* + (1 - 1' K^-1 k*)^2 / (1' K^-1 1),
    K = sf^2 R + diag(noise^2),  k* = sf^2 r*.

Plug-in means the hyperparameters are treated as known (no integration over them): this is the reference
point the protocol compares gcbml's full Bayes against, not a sampler. Cholesky solves only.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

SQRT5 = math.sqrt(5.0)
LOG_SF = (math.log(0.05), math.log(20.0))
LOG_ELL = (math.log(0.03), math.log(10.0))
LOG_NUG = (math.log(1e-7), math.log(1.0))


def matern52_ard(x1: np.ndarray, x2: np.ndarray, ell: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(((x1[:, None, :] - x2[None, :, :]) / ell) ** 2, 0.0).sum(-1))
    s = SQRT5 * d
    return (1.0 + s + s**2 / 3.0) * np.exp(-s)


def _chol_solves(K: np.ndarray, z: np.ndarray):
    cf = cho_factor(K, lower=True)
    one = np.ones(len(z))
    Ki1 = cho_solve(cf, one)
    denom = float(one @ Ki1)
    beta = float(one @ cho_solve(cf, z)) / denom
    return cf, Ki1, denom, beta


def _nll(theta, X, z, known_var, fit_nugget):
    sf, ell = math.exp(theta[0]), np.exp(theta[1:-1])
    nug = math.exp(theta[-1]) if fit_nugget else math.exp(LOG_NUG[0])
    K = sf**2 * matern52_ard(X, X, ell) + np.diag(known_var + nug**2)
    try:
        cf, _, _, beta = _chol_solves(K, z)
    except np.linalg.LinAlgError:
        return 1e25
    r = z - beta
    return 0.5 * float(r @ cho_solve(cf, r)) + float(np.log(np.diag(cf[0])).sum())


class PlugInGP:
    """Fit with ``PlugInGP.fit(X, y, known_sd=None, fit_nugget=True)``; predict with ``predict(Xs)``."""

    def __init__(self, X, y, known_sd, sf, ell, nugget, loglik):
        self.X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.ym, self.ys = float(y.mean()), float(y.std()) or 1.0
        self.z = (y - self.ym) / self.ys
        self.known_var = (
            np.zeros(len(y)) if known_sd is None else (np.asarray(known_sd, float) / self.ys) ** 2
        )
        self.sf, self.ell, self.nugget, self.loglik = sf, ell, nugget, loglik
        K = sf**2 * matern52_ard(self.X, self.X, ell) + np.diag(self.known_var + nugget**2)
        self._cf, self._Ki1, self._denom, self.beta = _chol_solves(K, self.z)
        self._alpha = cho_solve(self._cf, self.z - self.beta)

    @classmethod
    def fit(cls, X, y, known_sd=None, fit_nugget=True, n_starts=6, seed=0) -> PlugInGP:
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        ys = float(y.std()) or 1.0
        z = (y - y.mean()) / ys
        kv = np.zeros(n) if known_sd is None else (np.asarray(known_sd, float) / ys) ** 2
        rng = np.random.default_rng(seed)
        lo = np.array([LOG_SF[0]] + [LOG_ELL[0]] * d + [LOG_NUG[0]])
        hi = np.array([LOG_SF[1]] + [LOG_ELL[1]] * d + [LOG_NUG[1]])
        bounds = list(zip(lo, hi, strict=True))
        if not fit_nugget:
            bounds[-1] = (LOG_NUG[0], LOG_NUG[0] + 1e-9)
        best = None
        for k in range(n_starts):
            x0 = rng.uniform(lo, hi) if k else np.array([0.0] + [math.log(0.5)] * d + [math.log(1e-3)])
            r = minimize(
                _nll, np.clip(x0, lo, hi), args=(X, z, kv, fit_nugget), method="L-BFGS-B", bounds=bounds
            )
            if best is None or r.fun < best.fun:
                best = r
        th = best.x
        nug = math.exp(th[-1]) if fit_nugget else math.exp(LOG_NUG[0])
        return cls(X, y, known_sd, math.exp(th[0]), np.exp(th[1:-1]), nug, -float(best.fun))

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent mean and sd of f at Xs, in the units of y."""
        Xs = np.atleast_2d(np.asarray(Xs, float))
        ks = self.sf**2 * matern52_ard(self.X, Xs, self.ell)  # (n, m)
        mean = self.beta + ks.T @ self._alpha
        Kinv_ks = cho_solve(self._cf, ks)
        v = self.sf**2 - np.einsum("im,im->m", ks, Kinv_ks)
        v = v + (1.0 - self._Ki1 @ ks) ** 2 / self._denom
        return self.ym + self.ys * mean, self.ys * np.sqrt(np.maximum(v, 0.0))
