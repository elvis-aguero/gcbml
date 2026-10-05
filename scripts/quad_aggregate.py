"""Tables and the pass/fail verdict of item 3 from the ds_*.json of scripts/quad_vs_mcmc.py.

    uv run python scripts/quad_aggregate.py DIR

Pass rule (fixed by the senior engineer, not tuned): over all datasets, the median of |m_quad - m_mcmc| /
sigma_mcmc <= 0.25; the 90th percentile of sigma_quad / sigma_mcmc in [0.8, 1.25]; the coverage of the
true f0 by m +- 2 sigma not lower than the full-MCMC coverage by more than 3 points. "Over all datasets" is
read as the pool of all (dataset, x) pairs; the per-dataset medians are printed too.
"""

import json
import sys
from pathlib import Path

import numpy as np


def q(a, p):
    return float(np.percentile(a, p))


def main():
    res = [json.loads(f.read_text()) for f in sorted(Path(sys.argv[1]).glob("ds_*.json"), key=lambda f: int(f.stem[3:]))]
    print("(a) share of the full-MCMC h=0 variance explained by p (10 quantile bins, bias-corrected; raw in brackets)")
    print("id label                         lev  n  median-share  max-share   | rhat(p) rhat(m) it   mcmc_s   quad_s(warm) quad_s(1st)")
    allr, alls, cm, cq = [], [], [], []
    for r in res:
        sh, raw = np.array(r["p_share_corr"]), np.array(r["p_share_raw"])
        x = {k: np.array(v) for k, v in r["x"].items()}
        r["dm"] = np.abs(x["m_quad"] - x["m_mcmc"]) / x["s_mcmc"]
        r["sr"] = x["s_quad"] / x["s_mcmc"]
        r["cov_m"] = float(np.mean(np.abs(x["f0"] - x["m_mcmc"]) <= 2 * x["s_mcmc"]))
        r["cov_q"] = float(np.mean(np.abs(x["f0"] - x["m_quad"]) <= 2 * x["s_quad"]))
        m, qd = r["mcmc"], r["quad"]
        print(
            f"{r['id']:2d} {r['label']:30s} {r['n_levels']} {r['n']:3d}  {np.median(sh):6.3f} ({np.median(raw):.3f})"
            f"  {sh.max():6.3f} ({raw.max():.3f}) | {m['rhat_logp0']:.3f}  {m['rhat_mean']:.3f} {m['warmup']:4d} "
            f"{m['seconds_total']:8.0f} {qd['seconds_warm']:8.1f} {qd['seconds_first']:8.1f}"
        )
    print("\n(b) agreement and (c) coverage of f0 by m +- 2 sigma")
    print("id   |dm|/s med  p90   max   | s_q/s_m med  p90   max   | cov mcmc  cov quad | post sd log p: mcmc quad | ext | 40-pt d sigma max/med  d m/s max")
    for r in res:
        d, s, qd = r["dm"], r["sr"], r["quad"]
        pl = r["posterior_logp"]
        print(
            f"{r['id']:2d}   {np.median(d):6.3f} {q(d, 90):6.3f} {d.max():6.3f} | {np.median(s):6.3f} {q(s, 90):6.3f} {s.max():6.3f}"
            f" |  {r['cov_m']:.2f}      {r['cov_q']:.2f}    |  {pl['mcmc_sd']:.3f} {pl['quad_sd']:.3f} | {qd['n_extensions']} |"
            f" {qd['sigma_change_40']:.4f} / {qd['sigma_change_40_median']:.4f}   {qd['median_change_40_over_sigma']:.4f}"
        )
        allr.append(d)
        alls.append(s)
        cm.append(r["cov_m"])
        cq.append(r["cov_q"])
    D, S = np.concatenate(allr), np.concatenate(alls)
    covm, covq = float(np.mean(cm)), float(np.mean(cq))
    ok1, ok2, ok3 = np.median(D) <= 0.25, 0.8 <= q(S, 90) <= 1.25, covq >= covm - 0.03
    print("\nPooled over all (dataset, x):")
    print(f"  median |dm|/sigma = {np.median(D):.3f} (<= 0.25: {ok1}); p90 {q(D, 90):.3f}; max {D.max():.3f}")
    print(f"  p90 sigma_q/sigma_m = {q(S, 90):.3f} (in [0.8, 1.25]: {ok2}); median {np.median(S):.3f}; max {S.max():.3f}")
    print(f"  coverage mcmc {covm:.3f}, quad {covq:.3f}, diff {100 * (covq - covm):+.1f} points (>= -3: {ok3})")
    print(f"  median of per-dataset medians |dm|/sigma = {np.median([np.median(d) for d in allr]):.3f}")
    allsh = np.array([np.median(r["p_share_corr"]) for r in res])
    print(f"  p-share: median over datasets of median-over-x {np.median(allsh):.3f}; max over all x "
          f"{max(max(r['p_share_corr']) for r in res):.3f}")
    print(f"  grid doubling: max sigma change {max(r['quad']['sigma_change_40'] for r in res):.4f} (< 0.02 required)")
    print(f"  pass-1 extensions needed in {sum(r['quad']['n_extensions'] > 0 for r in res)} of {len(res)} datasets")
    print(f"VERDICT: {'ACCEPTED' if ok1 and ok2 and ok3 else 'REJECTED'}")


if __name__ == "__main__":
    main()
