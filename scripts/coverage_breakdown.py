"""Item 1 (speed2): coverage of the true f0 by m +- 2 sigma of the full MCMC, by dataset type and dataset.

    uv run python scripts/coverage_breakdown.py benchmarks/quad_vs_mcmc

Reads ds_*.json of scripts/quad_vs_mcmc.py (full-MCMC m, sigma and the true f0 at the 100 points of Sigma_N).
Types: A12 3 levels (ids 0-3), A12 4 levels (4-7), prior 3 levels (8-9), prior 4 levels (10-11).
Per dataset: z = (f0 - m) / sigma: mean, sd, share |z| < 2, share |z| < 1, max |z|.
Effective number of independent datasets: the datasets are independent, the 100 points of one dataset are
not, so the type-level coverage has at most (number of datasets) degrees of freedom. The script also prints the
effective number of independent points per dataset from the lag-1 structure of the z-scores along the Sobol
design is not meaningful, so it uses n_eff = n_points / (1 + (n_points - 1) rho) with rho the mean pairwise
correlation of |z| < 2 indicators across datasets of the same type is not estimable from 2-4 datasets: the
dataset count is the honest unit.
"""

import json
import sys
from pathlib import Path

import numpy as np

TYPES = {
    "A12 3-level": range(0, 4),
    "A12 4-level": range(4, 8),
    "prior 3-level": range(8, 10),
    "prior 4-level": range(10, 12),
}
res = {int(f.stem[3:]): json.loads(f.read_text()) for f in Path(sys.argv[1]).glob("ds_*.json")}
Z = {}
print("id  label                              n_x  mean z  sd z   cov2   cov1  max|z|  frac |z|>3")
for i in sorted(res):
    x = {k: np.array(v) for k, v in res[i]["x"].items()}
    z = (x["f0"] - x["m_mcmc"]) / x["s_mcmc"]
    Z[i] = z
    print(
        f"{i:2d}  {res[i]['label']:34s} {len(z):3d} {z.mean():7.2f} {z.std():5.2f} {np.mean(abs(z) < 2):6.2f} {np.mean(abs(z) < 1):6.2f} {abs(z).max():7.2f} {np.mean(abs(z) > 3):6.2f}"
    )
print("\ntype            datasets  coverage(2 sigma)  coverage(1 sigma)  sd of z  [per-dataset coverage]")
for name, ids in TYPES.items():
    z = np.concatenate([Z[i] for i in ids])
    per = [np.mean(abs(Z[i]) < 2) for i in ids]
    print(
        f"{name:14s} {len(ids):5d}     {np.mean(abs(z) < 2):8.3f}          {np.mean(abs(z) < 1):8.3f}        {z.std():5.2f}   {np.round(per, 2).tolist()}"
    )
allz = np.concatenate(list(Z.values()))
print(f"\nall: coverage {np.mean(abs(allz) < 2):.3f}")
# binomial error with the dataset count as the sample size: mean of per-dataset coverages, s.e. across datasets
for name, ids in {"A12": range(0, 8), "prior": range(8, 12)}.items():
    per = np.array([np.mean(abs(Z[i]) < 2) for i in ids])
    se = per.std(ddof=1) / np.sqrt(len(per))
    print(
        f"{name}: mean per-dataset coverage {per.mean():.3f} +- {se:.3f} (s.e. over {len(per)} datasets); "
        f"target 0.954; t = {(per.mean() - 0.954) / se:.2f}"
    )
# a Gaussian posterior would give share |z|<2 of 0.954 per point; the shares of the z-scores themselves
print(
    "\nFor reference: sigma = (q84 - q16)/2 of a Gaussian equals its sd, so 0.954 is the target at each point."
)
