"""Slice sampling of covariance hyperparameters with surrogate data (Murray & Adams 2010, arXiv 1006.0868).

Used for the hyperparameters theta of a latent Gaussian field f ~ N(0, Sigma_theta) (in gcbml: the
latent log-noise-variance field zeta and the varying-order field pi). Sampling theta with f fixed
mixes badly when the data are strong; the surrogate-data method reparametrises f through auxiliary
Gaussian "surrogate data" g ~ N(f, S_theta) so that theta and f move together.

TODO(W1-B): read the paper (Section 3, its algorithm for the surrogate-data slice sampler, and
Section 3.2 for choosing S_theta automatically when the likelihood factorises over sites) and
implement it with this interface:

surrogate_slice_step(key, theta, f, prior_cov, loglik, log_prior_theta, surrogate_var, width)
        -> (theta_new, f_new, n_evals)
    theta: (q,) unconstrained hyperparameters; f: (n,) latent field.
    prior_cov(theta) -> (n, n) covariance Sigma_theta (zero mean).
    loglik(f) -> scalar log-likelihood of the data given f.
    log_prior_theta(theta) -> scalar.
    surrogate_var(theta) -> (n,) diagonal of S_theta.
    width: (q,) slice widths for theta (one slice move along a random direction or per coordinate,
    as in the paper; document which).
Cholesky factorisations go through gcbml.linalg (factor / solve), never an explicit inverse.
"""

from __future__ import annotations


def surrogate_slice_step(key, theta, f, prior_cov, loglik, log_prior_theta, surrogate_var, width):
    raise NotImplementedError("W1-B")
