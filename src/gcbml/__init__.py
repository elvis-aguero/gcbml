"""gcbml: Grid-Convergent Bayesian Machine Learning.

Multi-fidelity Gaussian-process learning whose target is the converged answer (resolution h -> 0),
with a learned convergence order, full Bayesian uncertainty, and budget-aware sequential design.
It extends the KRR-LR-GPR framework of Yi et al. (2024, arXiv 2407.15110). See docs/spec.md.
"""

from gcbml import _config  # noqa: F401  (enables float64 in JAX)

__version__ = "0.1.0"
