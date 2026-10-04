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

TODO(W3-A): implement.

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


def build_sites(data, n_controls: int):
    raise NotImplementedError("W3-A")


def noise_var(m_s, b_s, zeta, sites, row_site, data_mask):
    raise NotImplementedError("W3-A")


def zeta_cov(log_sigma_z, log_ell_z, sites, site_mask):
    raise NotImplementedError("W3-A")


def log_prior(m_s, b_s, log_sigma_z, log_ell_z, scales):
    raise NotImplementedError("W3-A")
