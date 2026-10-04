"""Posterior sampling for one candidate structure (spec Section 2.5): the Gibbs sampler of Yi-h.

Given (mu, delta, beta) integrated out (model.log_marginal), the remaining unknowns are:

  global block theta (unconstrained vector, slice-sampled coordinate-wise, mcmc.slice):
      log sigma_mu, log ell_mu (d), c0 (k), c1 (k), log p0 (k),
      log sigma_delta (k), log ell_x (k, d), and log ell_h (k) [twy2] or logit gamma (k) [lb],
      m_s, b_s (k), log ell_v [only if within_run]
  latent noise field zeta (n_sites)          elliptical slice (mcmc.elliptical), prior noise.zeta_cov
  its hyperparameters (log sigma_z, log ell_z (d_u + k))   surrogate-data slice (mcmc.surrogate)
  varying order (only if cfg_varying_order and h_kernel == "twy2", spec 2.3):
      log p_j(x) = log p0_j + pi_j(x), pi_j a latent GP on the distinct unit x of the real rows,
      elliptical slice for pi_j, surrogate-data slice for (log sigma_pi_j, log ell_pi_j);
      PC prior with P(sigma_pi > 0.3) = 0.05 and P(ell < 0.1) = 0.05 (spec 2.5 table)
  censored outputs: model.censored_sweep

Priors: priors.PriorScales; c0, c1 ~ N(0, S_c^2); sigma_mu, ell_mu PC prior with sigma0 = S_mu;
sigma_delta, ell_x PC prior with sigma0 = S_delta; log ell_h ~ N(log 0.5, 1) on hbar units [assumption,
document]; logit gamma ~ logit-uniform; log p0 ~ N(log_p_mean, log_p_sd^2); ell_v PC-type as ell.

One Gibbs iteration = slice sweep over theta; ESS step for zeta; surrogate step for its hyperparameters;
(same two for each pi_j); one censored sweep. The state is a pytree; mcmc.chains.run_chains runs 4 chains
(vmap), warm-up adapts the slice widths. Starting points: drawn from the prior, then 200 iterations of
slice updates on theta only (state the rule).

TODO(W3-A): implement.

fit(key, data, z, bounds, cfg, scales, n_controls, n_warmup, n_samples, n_chains=4, varying_order=False)
        -> Posterior
    data: PaddedData; z: (n_pad,) Lambda(y) with censored rows set to their bound initially;
    bounds: (n_pad,) Lambda(y) of censored rows (unused elsewhere).
Posterior: NamedTuple with
    params: ModelParams with leading axes (n_chains, n_samples)   (constrained, ready for model.*)
    z: (n_chains, n_samples, n_pad) imputed outputs
    theta: dict name -> (n_chains, n_samples, ...) unconstrained draws (for diagnostics)
    diagnostics: mcmc.diagnostics.summary over every scalar of theta and log sigma_z
    n_evals: total log-likelihood evaluations
flatten(posterior) -> (ModelParams with one leading axis S, z (S, n_pad))     pools chains for prediction.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class Posterior(NamedTuple):
    params: Any
    z: Any
    theta: dict
    diagnostics: dict
    n_evals: Any


def fit(
    key,
    data,
    z,
    bounds,
    cfg,
    scales,
    n_controls: int,
    n_warmup: int,
    n_samples: int,
    n_chains: int = 4,
    varying_order: bool = False,
) -> Posterior:
    raise NotImplementedError("W3-A")


def flatten(posterior: Posterior):
    raise NotImplementedError("W3-A")
