"""Add the multi-start p-quadrature (quadrature.fit_quadrature(multistart=True)) to the ds_*.json of
scripts/quad_vs_mcmc.py, on the same data (the datasets are regenerated from their ids; the full-MCMC results
stay as saved).

    PYTHONPATH=tests:scripts:. uv run python scripts/quad_variant.py DIR [ids ...]

Adds r["quad_ms"] and r["x"]["m_quad_ms"], r["x"]["s_quad_ms"]; read them with
``scripts/quad_aggregate.py DIR _ms``.
"""

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

import gcbml  # noqa: F401
import quad_vs_mcmc as qv
from gcbml import quadrature

out = Path(sys.argv[1])
ids = [int(a) for a in sys.argv[2:]] or list(range(12))
for i in ids:
    f = out / f"ds_{i}.json"
    r = json.loads(f.read_text())
    data, zp, Xs, f0, label, nl = qv.make_dataset(i)
    kw = dict(key=jax.random.key(i), data=data, z=zp, bounds=zp, cfg=qv.CFG, scales=qv.SCALES, n_controls=2)
    t0 = time.perf_counter()
    q = quadrature.fit_quadrature(**kw, multistart=True)
    t1 = time.perf_counter() - t0
    m, v = quadrature.mu_moments(q, data, qv.CFG, Xs)
    med, sig, *_ = quadrature.summarize(q.weights, m, v)
    q40 = quadrature.fit_quadrature(**kw, n_grid2=40, multistart=True)
    m40, v40 = quadrature.mu_moments(q40, data, qv.CFG, Xs)
    med40, sig40, *_ = quadrature.summarize(q40.weights, m40, v40)
    lp = q.log_p[:, 0]
    r["quad_ms"] = dict(
        seconds_first=t1, seconds_warm=t1, n_extensions=q.pass1["n_extensions"],
        converged_frac=float(np.mean(q.converged)), log_p_range=[float(lp[0]), float(lp[-1])],
        weight_max=float(q.weights.max()), edge_weight=float(max(q.weights[0], q.weights[-1])),
        sigma_change_40=float(np.max(np.abs(sig40 / sig - 1))),
        sigma_change_40_median=float(np.median(np.abs(sig40 / sig - 1))),
        median_change_40_over_sigma=float(np.max(np.abs(med40 - med) / sig)),
    )  # fmt: skip
    r["x"]["m_quad_ms"], r["x"]["s_quad_ms"] = med.tolist(), sig.tolist()
    mean = float(np.sum(q.weights * lp))
    r["posterior_logp"]["quad_ms_sd"] = float(np.sqrt(np.sum(q.weights * (lp - mean) ** 2)))
    f.write_text(json.dumps(r, indent=1))
    d = np.abs(med - np.array(r["x"]["m_mcmc"])) / np.array(r["x"]["s_mcmc"])
    print(
        f"[{i}] ms: {t1:.1f}s median |dm|/s {np.median(d):.3f}, 40-pt d sigma {r['quad_ms']['sigma_change_40']:.4f}",
        flush=True,
    )
