"""Yi-h likelihood (spec 2.2-2.3): Lambda(y) = rho0 + rho1 mu + delta + e, with mu, delta and beta
integrated out exactly given (c, p, hyperparameters); S1 within-run correlation; censored outputs by
data augmentation; the Gibbs sampler over the remaining parameters.

TODO(W2-A): the interface is fixed before that wave starts.
"""
