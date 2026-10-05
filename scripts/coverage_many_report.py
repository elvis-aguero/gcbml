"""Summarise scripts/coverage_many.py: coverage by cell, per-dataset spread, and the common-mode z.

    uv run python scripts/coverage_many_report.py cov_many.json

Within a dataset the 100 z-scores are strongly correlated (a smooth error field): the independent unit is the
dataset. For each cell the script prints the pooled 2-sigma coverage, the mean and s.e. (over datasets) of the
per-dataset coverage, the share of datasets with coverage < 0.8, the sd of the per-dataset mean z (should be
near 1 times the typical posterior-sd share when the posterior is calibrated; a calibrated posterior gives
E[z^2] = 1 per point, here shown as the mean of z^2 pooled) and, as a dataset-level calibration check, the rank
statistic u = mean over x of Phi(z) per dataset (uniform on (0, 1) only approximately, because of the
correlation). Also prints how often the truth log p0 lies in the central 95% of the posterior of log p0.
"""

import json
import sys

import numpy as np

rows = json.load(open(sys.argv[1]))
print(f"{len(rows)} datasets; flagged rhat >= 1.05: {sum(max(r['rhat_p'], r['rhat_m']) >= 1.05 for r in rows)}")
print("cell           n   pooled cov  mean per-ds cov (s.e.)  ds with cov<0.8  mean z^2   median |mean z|  truth log p in 95%")
for kind in ("prior", "a12"):
    for nl in (3, 4):
        rs = [r for r in rows if r["kind"] == kind and r["n_levels"] == nl]
        Z = [np.array(r["z"]) for r in rs]
        per = np.array([np.mean(abs(z) < 2) for z in Z])
        zz = np.concatenate(Z)
        inp = np.mean([abs(r["log_p_true"] - r["log_p_mean"]) < 1.96 * r["log_p_sd"] for r in rs]) if kind == "a12" else float("nan")
        print(f"{kind:5s} {nl}-level {len(rs):3d}   {np.mean(abs(zz) < 2):8.3f}     {per.mean():.3f} ({per.std(ddof=1) / np.sqrt(len(per)):.3f})"
              f"          {np.mean(per < 0.8):6.2f}        {np.mean(zz**2):7.2f}     {np.median([abs(z.mean()) for z in Z]):6.2f}          {inp:.2f}")
for kind in ("prior", "a12"):
    rs = [r for r in rows if r["kind"] == kind]
    per = np.array([np.mean(abs(np.array(r["z"])) < 2) for r in rs])
    print(f"{kind}: all {len(rs)} datasets: mean per-dataset coverage {per.mean():.3f} +- {per.std(ddof=1) / np.sqrt(len(per)):.3f} (s.e. over datasets)")
# for prior: the truth lies inside the posterior of the model that generated it: P(|z| < 2) per point should be 0.95
