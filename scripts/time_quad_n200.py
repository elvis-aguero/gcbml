"""Wall time of p-quadrature (literal and multistart) on the n ~ 200 A12 dataset, warm (second call).

PYTHONPATH=tests:scripts:. uv run python scripts/time_quad_n200.py [N0_PER_CONTROL]
"""

import sys
import time

import jax

import gcbml  # noqa: F401
import quad_vs_mcmc_helpers as h
from gcbml import model, quadrature

data, zp = h.big_dataset(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
cfg = model.ModelConfig()
print("n", data.n, "n_pad", data.X.shape[0], flush=True)
for ms in (False, True):
    for rep in range(2):
        t = time.perf_counter()
        q = quadrature.fit_quadrature(jax.random.key(0), data, zp, zp, cfg, h.SCALES, 2, multistart=ms)
        dt = time.perf_counter() - t
        print(
            f"multistart={ms} call {rep}: {dt:.1f} s, evals {q.n_evals}, converged {q.converged.mean():.2f}",
            flush=True,
        )
