"""One truth of the A12 test (spec Step 1): acquisition candidates against full-refit oracle values.

    uv run python -m benchmarks.a12.run_truth --seed 3 --out results/a12

For truth ``seed`` (A12Truth, d = 2): initial design at levels 0-2, then the acquisition's best candidate at
EACH level 0-4 (the overall best is one of them). The oracle value of each candidate is the expected
reduction of H from full MCMC refits on fantasy outcomes, with enough fantasies that the Monte Carlo s.e. of
the mean gain is below ``--se-target`` of it (or ``--n-max`` fantasies). The reference H is the mean over
``--n-base`` refits of the unchanged data, so the gain has no bias from the refit's own Monte Carlo error.
Writes <out>/truth_<seed>.json. No gates are evaluated (A12 is about the acquisition).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import gcbml  # noqa: F401
from gcbml import acquisition as acq
from gcbml.campaign import (
    Campaign,
    CampaignSettings,
    fantasy_dataset,  # noqa: F401
    oracle_values,
    refit_H,
)
from gcbml.cost import CostPrior
from gcbml.gates import GateResult
from gcbml.priors import PriorScales
from gcbml.synthetic import A12Truth

SCALES = PriorScales(S_mu=1.0, S_c=0.5, S_delta=0.5, S_noise=0.02)
COST_PRIOR = CostPrior(k0_mean=0.0, k0_sd=1.0, gamma_mean=(3.0,), gamma_sd=(0.5,))


class A12Campaign(Campaign):
    def _gates(self, an, key):  # A12 judges the acquisition, not the calibration gates
        return [GateResult(f"G{i}", "not testable", {}) for i in range(8)], None


def run(args) -> dict:
    t_start = time.time()
    truth = A12Truth(args.seed, d=2, budget=1e9, eps_abs=args.eps)
    settings = CampaignSettings(
        n_warmup=args.warmup,
        n_samples=args.samples,
        n_chains=args.chains,
        q=1,
        n_candidates_u=args.n_u,
        extra_levels=2,
        h_kernels=("twy2",),
        max_draws_acquisition=args.max_draws,
        seed=args.seed + 1000,
    )
    c = A12Campaign(truth.problem(), SCALES, COST_PRIOR, settings)
    probes = c.initial_design()
    o = truth.oracle(seed=args.seed)
    o.submit(probes, [c.cap_of(p) for p in probes])
    c.tell(o.poll())
    an = c._analyse()
    t_fit = time.time() - t_start
    plan = c.plan(key=jax.random.PRNGKey(args.seed))
    tab = plan.table
    ok = np.asarray(tab.admissible) & (np.asarray(tab.gain) > 0)
    levels = np.array([int(cd.levels[0]) for cd in plan.cands])
    chosen = []
    for lev in range(5):
        idx = np.flatnonzero(ok & (levels == lev))
        if idx.size:
            chosen.append(int(idx[np.argmax(np.asarray(tab.ratio)[idx])]))
    t_plan = time.time() - t_start
    # unpaired mode only: reference H = mean of refits of the unchanged data
    hs, rh0 = [], []
    for b in range(0 if args.paired else args.n_base):
        h, rh = refit_H(c, an, c.dataset, jax.random.PRNGKey(10_000 + b))
        hs.append(h)
        rh0.append(rh)
    h_base = float(np.mean(hs)) if hs else float(acq.H_value(an.sigma_epi, an.eps, c.mode))
    cands = [plan.cands[i] for i in chosen]
    ranking = oracle_values(
        truth, c, cands, args.n_min, key=jax.random.PRNGKey(args.seed + 5),
        se_target=args.se_target, n_max=args.n_max, H_base=h_base, warm=args.warm, paired=args.paired,
        refit_warmup=args.refit_warmup,
    )  # fmt: skip
    by_idx = {r.index: r for r in ranking}
    rows = []
    for k, i in enumerate(chosen):
        r = by_idx[k]
        rows.append(
            dict(
                level=int(levels[i]), site=[float(x) for x in np.asarray(cands[k].Xa)[0]],
                cost_mean=float(cands[k].cost_mean), acq_gain=float(tab.gain[i]),
                acq_ratio=float(tab.ratio[i]), oracle_gain=r.gain, oracle_value=r.value,
                oracle_gain_se=r.se * float(cands[k].cost_mean), n_fantasies=r.n, rel_se=r.rel_se,
                rhat_p0=r.rhat_p0, raw_gains=list(r.raw_gains), raw_rhats=list(r.raw_rhats),
                n_discarded=r.n_discarded,
            )
        )  # fmt: skip
    out = dict(
        seed=args.seed, p=truth.p, H_now=float(acq.H_value(an.sigma_epi, an.eps, c.mode)), H_base=h_base,
        H_base_sd=float(np.std(hs)) if hs else 0.0, rhat_p0_base=max(rh0) if rh0 else float('nan'),
        paired=args.paired, warm=args.warm, rhat_p0_data=None, candidates=rows,
        n_u=args.n_u, settings=dict(warmup=args.warmup, samples=args.samples, chains=args.chains),
        seconds=dict(fit=t_fit, plan=t_plan - t_fit, total=time.time() - t_start),
    )  # fmt: skip
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results/a12")
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n-u", type=int, default=16)
    ap.add_argument("--max-draws", type=int, default=64)
    ap.add_argument("--n-base", type=int, default=3)
    ap.add_argument("--refit-warmup", type=int, default=150)
    ap.add_argument("--paired", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--n-min", type=int, default=12)
    ap.add_argument("--n-max", type=int, default=128)
    ap.add_argument("--se-target", type=float, default=0.1)
    args = ap.parse_args(argv)
    res = run(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"truth_{args.seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["seconds"]), flush=True)


if __name__ == "__main__":
    main()
