"""Gradient-free MCMC primitives (spec Section 2.5).

slice.py        univariate slice sampling with stepping out and shrinkage, applied coordinate-wise
                (Neal 2003, arXiv physics/0009028, Figs. 3 and 5).
elliptical.py   elliptical slice sampling for a latent field with a zero-mean Gaussian prior
                (Murray, Adams & MacKay 2010, arXiv 1001.0175, Fig. 2).
surrogate.py    slice sampling of covariance hyperparameters with surrogate data
                (Murray & Adams 2010, arXiv 1006.0868).
chains.py       run several chains with jax.vmap and lax.scan; warm-up adapts the slice widths.
diagnostics.py  rank-normalised split R-hat, folded R-hat, bulk-ESS and tail-ESS
                (Vehtari et al. 2021, arXiv 1903.08008, Sections 3-4).

All samplers take a jax PRNG key and pure log-density callables, are jit-compatible, and return
the number of log-density evaluations used (for cost accounting).
"""
