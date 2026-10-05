"""Item 1 (speed2): coverage with the TRUE parameters of prior-draw datasets (no MCMC).

    PYTHONPATH=tests:scripts:. uv run python scripts/coverage_true_params.py N_PER_CELL

For a dataset drawn from the model's own prior, conditioning on the true (theta, zeta) the posterior of mu(x)
given the data is exactly Gaussian (model.predict_mu), so z = (f0 - mean) / sd is N(0, 1) at every x and
E z^2 = 1. This separates the simulator, the likelihood and predict_mu from the sampler: if z is not N(0, 1)
here, the data are not drawn from the model that the code evaluates.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

import gcbml  # noqa: F401
import quad_vs_mcmc as qv
from gcbml import inference, model

n_cell = int(sys.argv[1])
for nl in (3, 4):
    zs, per = [], []
    for k in range(n_cell):
        i = 300 + k
        data, zp, Xs, f0, label, _ = qv.make_dataset(i, "prior", nl)
        smp = inference._Sampler(data, zp, zp, qv.CFG, qv.SCALES, 2, False)
        p = smp.constrain(qv.LAST_ST)
        m, v = model.predict_mu(p, data, jnp.asarray(zp), qv.CFG, jnp.asarray(Xs))
        z = (f0 - np.asarray(m)) / np.sqrt(np.asarray(v))
        zs.append(z)
        per.append(np.mean(abs(z) < 2))
    Z = np.concatenate(zs)
    mz = np.array([z.mean() for z in zs])
    print(f"{nl}-level: {n_cell} datasets, pooled coverage {np.mean(abs(Z) < 2):.3f}, per-dataset mean {np.mean(per):.3f} "
          f"(s.e. {np.std(per, ddof=1) / np.sqrt(n_cell):.3f}), E z^2 {np.mean(Z**2):.2f}, sd of per-dataset mean z {mz.std():.2f}, "
          f"datasets with cov < 0.8: {np.mean(np.array(per) < 0.8):.2f}, max|z| {abs(Z).max():.1f}")
