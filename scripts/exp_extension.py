"""Item 6: does fit()'s automatic extension fix a dataset whose chains had not converged? usage: exp_extension.py ID"""

import sys
import time

import jax

sys.path.insert(0, "tests")
sys.path.insert(0, "scripts")
import quad_vs_mcmc as qv  # noqa: E402

from gcbml import inference  # noqa: E402

i = int(sys.argv[1])
data, zp, Xs, f0, label, nl = qv.make_dataset(i, "prior", 4)
for ext in (0, 2):
    t = time.perf_counter()
    post = inference.fit(jax.random.key(i), data, zp, zp, qv.CFG, qv.SCALES, 2, 500, 500, 4, max_extensions=ext)
    d = post.diagnostics
    print(
        f"id {i} max_extensions={ext}: used {d.n_extensions}, max rhat {max(v[0] for v in d.values()):.3f}, "
        f"min bulk ESS {min(v[1] for v in d.values()):.0f}, {time.perf_counter() - t:.0f}s",
        flush=True,
    )
