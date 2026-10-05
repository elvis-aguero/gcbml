"""Does a paired (common random numbers), warm-started refit cut the Monte Carlo error of the gain?

    uv run python -m benchmarks.a12.paired_probe --seed 0 --out results/a12_paired

On one truth, after the initial design, for ONE candidate (the acquisition's best at --level) and F fixed
fantasy outcomes, the gain H_ref - H_after is estimated K times with different MCMC keys by
  cold:    cold-start refit, independent keys, H_ref = fixed mean of base refits (as before);
  warm:    warm-started refit (--refit-warmup iterations), independent keys, same fixed H_ref;
  paired:  warm-started refits of the unchanged data and of the data + fantasy with the SAME key.
The s.e. of the gain at a fixed fantasy is the sd over the K keys; the table gives its mean over fantasies and
the factor cold/warm and cold/paired, with seconds per refit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import gcbml  # noqa: F401
from benchmarks.a12.run_truth import COST_PRIOR, SCALES, A12Campaign
from gcbml.campaign import CampaignSettings, _fantasy_outcome, fantasy_dataset, refit_H
from gcbml.synthetic import A12Truth


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--n-fantasies", type=int, default=4)
    ap.add_argument("--n-keys", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--refit-warmup", type=int, default=150)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--out", default="results/a12_paired")
    a = ap.parse_args(argv)
    truth = A12Truth(a.seed, d=2, budget=1e9, eps_abs=0.05)
    s = CampaignSettings(
        n_warmup=a.warmup, n_samples=a.samples, n_chains=4, q=1, n_candidates_u=16, extra_levels=2,
        h_kernels=("twy2",), max_draws_acquisition=64, seed=a.seed + 1000,
    )  # fmt: skip
    c = A12Campaign(truth.problem(), SCALES, COST_PRIOR, s)
    probes = c.initial_design()
    o = truth.oracle(seed=a.seed)
    o.submit(probes, [c.cap_of(p) for p in probes])
    c.tell(o.poll())
    an = c._analyse()
    plan = c.plan(key=jax.random.PRNGKey(a.seed))
    levels = np.array([int(cd.levels[0]) for cd in plan.cands])
    ok = np.asarray(plan.table.admissible) & (np.asarray(plan.table.gain) > 0) & (levels == a.level)
    ci = int(np.flatnonzero(ok)[np.argmax(np.asarray(plan.table.ratio)[ok])])
    cand = plan.cands[ci]
    ys = [
        _fantasy_outcome(truth, c, an, cand, jax.random.PRNGKey(500 + j), "model")
        for j in range(a.n_fantasies)
    ]
    dss = [fantasy_dataset(c, cand, y, f"p{j}") for j, y in enumerate(ys)]
    secs = {"cold": [], "warm": [], "paired_ref": [], "paired_fant": []}
    # fixed reference H for the unpaired variants: mean of a few cold refits of the unchanged data
    h_ref = float(np.mean([refit_H(c, an, c.dataset, jax.random.PRNGKey(9000 + b))[0] for b in range(3)]))
    gains = {"cold": [], "warm": [], "paired": []}  # [fantasy][key]
    for dsj in dss:
        for name in ("cold", "warm"):
            row = []
            for k in range(a.n_keys):
                t0 = time.time()
                h, _ = refit_H(c, an, dsj, jax.random.PRNGKey(100 + k), name == "warm", a.refit_warmup)
                secs[name].append(time.time() - t0)
                row.append(h_ref - h)
            gains[name].append(row)
    refs = []
    for k in range(a.n_keys):
        t0 = time.time()
        refs.append(refit_H(c, an, c.dataset, jax.random.PRNGKey(100 + k), True, a.refit_warmup)[0])
        secs["paired_ref"].append(time.time() - t0)
    for dsj in dss:
        row = []
        for k in range(a.n_keys):
            t0 = time.time()
            h, _ = refit_H(c, an, dsj, jax.random.PRNGKey(100 + k), True, a.refit_warmup)
            secs["paired_fant"].append(time.time() - t0)
            row.append(refs[k] - h)
        gains["paired"].append(row)
    sd = {k: float(np.mean([np.std(r, ddof=1) for r in v])) for k, v in gains.items()}
    mean_gain = {k: float(np.mean(v)) for k, v in gains.items()}
    out = dict(
        seed=a.seed, level=a.level, h_ref=h_ref, sd_gain_at_fixed_fantasy=sd, mean_gain=mean_gain,
        factor_warm=sd["cold"] / sd["warm"], factor_paired=sd["cold"] / sd["paired"],
        seconds_per_refit={k: float(np.mean(v)) for k, v in secs.items()}, gains=gains,
        settings=dict(warmup=a.warmup, refit_warmup=a.refit_warmup, samples=a.samples),
    )  # fmt: skip
    p = Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"paired_{a.seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "gains"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
