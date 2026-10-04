"""Heteroscedastic run-to-run noise (spec Section 2.4).

    log s^2(x, h) = m_s + sum_j b_sj hbar_j + zeta(site),     zeta ~ GP(0, sigma_z^2 matern52 over (u, hbar))

A *site* is a distinct (u, h) pair: replicate runs share a site, and the spread among them is what
identifies s^2. zeta lives on the sites (n_sites values), not on the output rows. Row i of the data
gets s^2 of its run's site. The output coordinates v (module S1) do not enter the noise variance; the
correlation of outputs within one run is model.ModelConfig.within_run (ell_v).

Priors (spec 2.5 table, gcbml.priors.PriorScales): m_s ~ N(2 log S_noise, log(10)^2);
b_sj ~ N(0, b_s_sd^2) (symmetric: the spread may grow or shrink with h, spec 2.4);
(sigma_z, ell_z) PC prior (priors.pc_matern_logpdf, per ARD length scale) with sigma0 = 1 (log-variance
units: a factor e in the variance) and ell0 = PriorScales.ell0.

Implementation notes. Site extraction runs outside jit (NumPy, exact float equality on (u, hbar)); the
padded site rows copy site 0, so every array stays finite. Sites are numbered in order of first
appearance. zeta_cov has the identity on the padded block, so a Cholesky factor of it exists and the
padded zeta values (which no row reads) are independent N(0, 1) draws that never enter a likelihood.

build_sites(data, n_controls) -> (sites (n_sites_pad, n_controls + k), row_site (n_pad,), site_mask)
    Unique (u, hbar) pairs of the real rows of PaddedData (u = the first n_controls columns of X),
    padded to a bucket size; row_site[i] is the site index of row i (0 for padded rows).
noise_var(m_s, b_s, zeta, sites, row_site, data_mask) -> (n_pad,)
    s^2 per row = exp(m_s + sites[row_site, n_controls:] @ b_s + zeta[row_site]); padded rows get 1.
    (n_controls is implied by b_s.shape: hbar is the last k columns of sites.)
zeta_cov(log_sigma_z, log_ell_z, sites, site_mask) -> (n_sites_pad, n_sites_pad)
    sigma_z^2 * ard_matern(sites, sites, ell_z, 2.5), padded block decoupled as in linalg.
log_prior(m_s, b_s, log_sigma_z, log_ell_z, scales) -> scalar
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from gcbml._config import bucket
from gcbml.kernels import ard_matern
from gcbml.priors import PriorScales, normal_logpdf, pc_matern_logpdf


def unique_rows(A, mask):
    """Distinct rows of the real rows of A (first-appearance order), padded to a bucket size.

    Returns (rows (m_pad, c), row_index (n,), row_mask (m_pad,)). Padded rows of ``rows`` copy row 0 and
    ``row_index`` is 0 on masked rows. Exact equality (replicates repeat the same floats).
    """
    A = np.asarray(A, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = A.shape[0]
    index = np.zeros(n, dtype=np.int64)
    seen: dict[tuple, int] = {}
    reps: list[int] = []
    for i in np.flatnonzero(mask):
        key = tuple(A[i])
        if key not in seen:
            seen[key] = len(reps)
            reps.append(i)
        index[i] = seen[key]
    m = len(reps)
    if m == 0:
        raise ValueError("no real rows")
    m_pad = bucket(m)
    rows = np.repeat(A[reps[0]][None, :], m_pad, axis=0)
    rows[:m] = A[reps]
    row_mask = np.zeros(m_pad, dtype=bool)
    row_mask[:m] = True
    return rows, index, row_mask


def build_sites(data, n_controls: int):
    A = np.hstack([np.asarray(data.X, dtype=float)[:, :n_controls], np.asarray(data.H, dtype=float)])
    return unique_rows(A, data.mask)


def noise_var(m_s, b_s, zeta, sites, row_site, data_mask):
    b_s = jnp.asarray(b_s, dtype=float)
    sites = jnp.asarray(sites, dtype=float)
    k = b_s.shape[0]
    log_s2_site = m_s + sites[:, sites.shape[1] - k :] @ b_s + jnp.asarray(zeta, dtype=float)
    s2 = jnp.exp(log_s2_site)[jnp.asarray(row_site)]
    return jnp.where(jnp.asarray(data_mask, dtype=bool), s2, 1.0)


def zeta_cov(log_sigma_z, log_ell_z, sites, site_mask):
    sites = jnp.asarray(sites, dtype=float)
    site_mask = jnp.asarray(site_mask, dtype=bool)
    K = jnp.exp(2.0 * log_sigma_z) * ard_matern(sites, sites, jnp.exp(log_ell_z), 2.5)
    both = site_mask[:, None] & site_mask[None, :]
    return jnp.where(both, K, 0.0) + jnp.diag(jnp.where(site_mask, 0.0, 1.0))


def log_prior(m_s, b_s, log_sigma_z, log_ell_z, scales: PriorScales):
    return (
        normal_logpdf(m_s, 2.0 * math.log(scales.S_noise), math.log(10.0))
        + normal_logpdf(b_s, 0.0, scales.b_s_sd)
        + pc_matern_logpdf(log_sigma_z, log_ell_z, 1.0, scales.ell0)
    )
