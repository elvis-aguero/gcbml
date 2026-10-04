"""Cost model (spec Section 2.6), learned from the costs the oracle reports.

For run r with unit controls u_r (n_controls,), integer level vector l_r (k,) (0 = coarsest) and an
optional oracle quote q_r (core-hours; None -> no quote):

    log2 c_r = kappa0 + sum_j gamma_j l_rj + a * has_q_r * (log2 q_r - qbar) + omega(u_r, l_r) + eta_r
    omega ~ GP(0, sigma_w^2 ard_matern52 over (u, l / l_scale)),   eta_r ~ N(0, s_eta^2)

qbar is the mean of log2 q over runs with a quote (centring). The power law 2^{gamma l} is only the
prior mean: omega lets the posterior depart from it (spec 2.6; the user requires an expressive posterior).
A run stopped at its cap has a right-censored cost: log2 c_r >= log2 cap_r (Tobit; handled by data
augmentation, like model.censored_sweep).

Priors (problem-specific scales are required inputs, CostPrior): kappa0 ~ N(k0_mean, k0_sd^2);
gamma_j ~ N(gamma_mean_j, gamma_sd_j^2); a ~ N(1, 0.5^2) (a quote is informative but not trusted);
(sigma_w, ell_w) PC prior with sigma0 = 1 (log2 units), ell0 = 0.1; s_eta: PC/exponential with
P(s_eta > 1) = 0.05.

Inference: given (sigma_w, ell_w, s_eta) and the imputed censored values, log2 c is Gaussian with the
linear coefficients (kappa0, gamma, a) Gaussian: integrate them and omega out exactly (same algebra as
model.py with a Gaussian beta). Slice-sample the 2 + d_u + k hyperparameters (mcmc.slice), and Gibbs the
censored values (truncated normal full conditionals). Use gcbml.linalg for every solve.

TODO(W3-B): implement.

CostData: NamedTuple(U (n_pad, n_controls) unit, L (n_pad, k) float levels, log2c (n_pad,),
                     censored (n_pad,) bool, log2q (n_pad,) (0 where no quote), has_q (n_pad,) bool,
                     mask (n_pad,) bool)
CostPrior: dataclass(k0_mean, k0_sd, gamma_mean (k,), gamma_sd (k,), l_scale=4.0)
fit_cost(key, data, prior, n_warmup, n_samples, n_chains=4) -> CostPosterior (draws of hyperparameters and
    imputed log2c; diagnostics as in mcmc.diagnostics.summary)
predict_log2(post, U_new, L_new, log2q_new, has_q_new) -> (mean (S, m), var (S, m))
    Predictive of log2 c for new runs per posterior draw (includes eta).
expected_cost(mean, var, w) -> (m,)       E[c] = weighted average of 2^mean * exp(0.5 (ln 2)^2 var).
cost_cap(mean, var, w, q=0.95) -> (m,)     quantile q of the pooled predictive of c (mixture over draws;
    solve the mixture CDF by bisection in log2 space).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


class CostData(NamedTuple):
    U: Any
    L: Any
    log2c: Any
    censored: Any
    log2q: Any
    has_q: Any
    mask: Any


@dataclass(frozen=True)
class CostPrior:
    k0_mean: float
    k0_sd: float
    gamma_mean: tuple[float, ...]
    gamma_sd: tuple[float, ...]
    l_scale: float = 4.0


class CostPosterior(NamedTuple):
    hyper: dict
    log2c: Any
    diagnostics: dict
    data: CostData
    prior: CostPrior


def fit_cost(
    key, data: CostData, prior: CostPrior, n_warmup: int, n_samples: int, n_chains: int = 4
) -> CostPosterior:
    raise NotImplementedError("W3-B")


def predict_log2(post: CostPosterior, U_new, L_new, log2q_new, has_q_new):
    raise NotImplementedError("W3-B")


def expected_cost(mean, var, w):
    raise NotImplementedError("W3-B")


def cost_cap(mean, var, w, q: float = 0.95):
    raise NotImplementedError("W3-B")
