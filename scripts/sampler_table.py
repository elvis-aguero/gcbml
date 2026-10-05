"""Table of scripts/bench_sampler.py results.

uv run python scripts/sampler_table.py DIR [NAME ...]
"""

import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
names = sys.argv[2:] or sorted(p.stem for p in d.glob("*.json"))
print(
    "dataset      n  sampler  wall_s  evals(lik|grad+lik)   ESS: log_p0 log_sig_mu mu[a] mu[b] mu[c]  min-ESS/s  min-ESS/1000ev  rhat_max  div  cov_f0"
)
for n in names:
    r = json.loads((d / f"{n}.json").read_text())
    for k in ("gibbs", "nuts"):
        if k not in r:
            continue
        m = r[k]
        div = sum(r["nuts_info"]["divergences"]) if k == "nuts" else "-"
        print(
            f"{n:11s} {r['n']:3d}  {k:6s} {m['seconds']:7.1f}  {m['evals']:>9d}   "
            + " ".join(f"{e:6.0f}" for e in m["ess"].values())
            + f"   {m['ess_min_per_s']:8.2f}  {m['ess_min_per_1000_evals']:8.2f}  {m['rhat_max_all']:8.3f}  {div!s:>4}  {m.get('coverage_f0', float('nan')):.2f}"
        )
    if "agreement" in r:
        a = r["agreement"]
        print(
            f"{'':11s}      agreement: KS(log p0) {a['ks_log_p0']:.3f}  median|dm|/sigma {a['median_dm_over_sigma']:.3f}  p90 {a['p90_dm_over_sigma']:.3f}  sigma ratio {a['sigma_ratio_median']:.2f}"
        )
