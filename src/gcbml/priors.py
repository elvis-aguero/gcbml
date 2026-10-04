"""Prior densities and constraining maps (spec Section 2.5).

MCMC runs on an unconstrained real vector. Positive parameters use exp, (0,1) parameters use the
logistic map, and every log density returned here is the density of the UNCONSTRAINED variable
(it includes the log-Jacobian of the map). No default here is tied to an application: physical
scales (S_mu, S_c, S_delta, S_noise) are required inputs; defaults exist only for quantities that
are generic in unit coordinates (length scales) or dimensionless (orders).


pc_matern_logpdf(log_sigma, log_ell, sigma0, ell0, alpha_sigma=0.05, alpha_ell=0.05)
    PC prior for the (sigma, ell) of a Matérn field with d = 1 (Fuglstad et al. 1503.00256, Thm 2.6),
    applied per ARD length scale (a heuristic; spec 2.5). Calibrated by P(sigma > sigma0) = alpha_sigma
    and P(ell < ell0) = alpha_ell. Read Thm 2.6 and write the d = 1 density in (sigma, ell); note that
    a range parameter rho = c * ell gives the same density in ell for any constant c, because the
    calibration event {rho < c ell0} is the same as {ell < ell0}. For ARD (log_ell of shape (d,)), sum
    the log density of the d independent length-scale factors and add the sigma factor once.
    Return the log density of (log_sigma, log_ell) (include the exp Jacobians).
lognormal_logpdf(log_x, mu, sd)       density of log_x ~ N(mu, sd) (i.e. x lognormal).
normal_logpdf(x, mean, sd)
logit_uniform_logpdf(logit_x)         x = sigmoid(logit_x) ~ Uniform(0, 1): density of logit_x.
Each function must sum over trailing array dimensions and return a scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import log_sigmoid


@dataclass(frozen=True)
class PriorScales:
    """Problem-specific prior scales, all in Lambda units (spec 2.5, table). Required, no defaults."""

    S_mu: float  # sd of the converged value mu(x) around its mean
    S_c: float  # sd of the transfer coefficients c_0j, c_1j (size of the coarsest-level error)
    S_delta: float  # P(sigma_delta > S_delta) = 0.05
    S_noise: float  # typical run-to-run sd; m_s ~ N(2 log S_noise, log(10)^2)
    ell0: float = 0.1  # P(ell < ell0) = 0.05, unit coordinates (generic)
    log_p_mean: float = 0.0  # log p0 ~ N(log_p_mean, log_p_sd^2): median order 1
    log_p_sd: float = 1.0
    b_s_sd: float = math.log(4.0)  # noise trend in hbar, symmetric (spec 2.4)


def pc_matern_logpdf(log_sigma, log_ell, sigma0, ell0, alpha_sigma=0.05, alpha_ell=0.05):
    # Thm 2.6 of 1503.00256 with d = 1: pi(sigma, rho) = 0.5 l1 l2 rho^{-3/2} exp(-l1 rho^{-1/2} - l2 sigma),
    # l1 = -log(alpha_ell) rho0^{1/2}, l2 = -log(alpha_sigma) / sigma0. The density factorises, so with
    # u = log sigma, v = log ell and the Jacobians exp(u), exp(v):
    #   sigma factor: log l2 + u - l2 e^u        (one per call)
    #   ell factor:   log(0.5 l1) - v/2 - l1 e^{-v/2}   (one per length scale)
    log_sigma = jnp.asarray(log_sigma, dtype=float)
    log_ell = jnp.asarray(log_ell, dtype=float)
    l1 = -math.log(alpha_ell) * jnp.sqrt(ell0)
    l2 = -math.log(alpha_sigma) / sigma0
    sigma_part = jnp.log(l2) + log_sigma - l2 * jnp.exp(log_sigma)
    ell_part = jnp.log(0.5 * l1) - 0.5 * log_ell - l1 * jnp.exp(-0.5 * log_ell)
    return jnp.sum(sigma_part) + jnp.sum(ell_part)


def lognormal_logpdf(log_x, mu, sd):
    return normal_logpdf(log_x, mu, sd)


def normal_logpdf(x, mean, sd):
    x = jnp.asarray(x, dtype=float)
    z = (x - mean) / sd
    return jnp.sum(-0.5 * z**2 - jnp.log(sd) - 0.5 * math.log(2.0 * math.pi))


def logit_uniform_logpdf(logit_x):
    # x = sigmoid(t) ~ U(0,1); dx/dt = x (1 - x) = sigmoid(t) sigmoid(-t).
    t = jnp.asarray(logit_x, dtype=float)
    return jnp.sum(log_sigmoid(t) + log_sigmoid(-t))
