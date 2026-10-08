"""Aggregate the A12 truths (spec Step 1): regret criterion (i) and per-level calibration (ii).

    uv run python -m benchmarks.a12.aggregate_a12 results/a12 [--n 20]

Criterion (i): per truth, the oracle value per cost of the acquisition's choice (the candidate with the best
acquisition ratio among the five per-level bests) is >= 0.8 x the best oracle value per cost; needed in at
least 16 of 20 truths. Criterion (ii): for every level, the median over truths of (acquisition gain / oracle
gain) lies in [0.5, 2]. A candidate with no valid fantasy (all dropped by the rhat and divergence rules, or
a non-finite value) is EXCLUDED from the oracle's best and from the per-level ratio, and counted per truth;
if the acquisition's own choice has no valid fantasy the truth counts as a miss. Exit code 0 only if both
criteria hold on all ``--n`` truths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def valid(x) -> bool:
    return x["n_fantasies"] > 0 and np.isfinite(x["oracle_value"]) and np.isfinite(x["oracle_gain"])


def aggregate(res: list, need: int = 16, n: int = 20) -> dict:
    rows, hits, ratios, excluded, none = [], 0, {lev: [] for lev in range(5)}, {}, []
    for r in res:
        c = r["candidates"]
        if not c:  # no candidate had a positive acquisition gain at any level: a miss
            none.append(r["seed"])
            excluded[r["seed"]] = 0
            rows.append((r, None, None, float("nan"), False, []))
            continue
        excluded[r["seed"]] = sum(not valid(x) for x in c)
        acq_i = int(np.argmax([x["acq_ratio"] for x in c]))
        good = [i for i, x in enumerate(c) if valid(x)]
        orc_i = max(good, key=lambda i: c[i]["oracle_value"]) if good else None
        if orc_i is None or not valid(c[acq_i]):
            regret, hit = float("nan"), False
        else:
            best = c[orc_i]["oracle_value"]
            regret = c[acq_i]["oracle_value"] / best if best > 0 else float("nan")
            hit = bool(regret >= 0.8) if best > 0 else True
        hits += hit
        cells = []
        for x in c:
            g = x["acq_gain"] / x["oracle_gain"] if valid(x) and x["oracle_gain"] > 0 else float("nan")
            ratios[x["level"]].append(g)
            cells.append((x["level"], g, x["rel_se"], x["n_fantasies"], x.get("n_discarded", 0)))
        rows.append((r, c[acq_i]["level"], None if orc_i is None else c[orc_i]["level"], regret, hit, cells))
    return dict(rows=rows, hits=hits, ratios=ratios, excluded=excluded, no_candidates=none, need=need, n=n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--need", type=int, default=16)
    args = ap.parse_args(argv)
    files = sorted(Path(args.dir).glob("truth_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    res = [json.loads(f.read_text()) for f in files]
    out = aggregate(res, args.need, args.n)
    print(f"{len(res)} of {args.n} truths found")
    print(
        "seed   p   choice_lvl oracle_best_lvl regret  excluded | gain acq/oracle per level 0..4 "
        "(oracle rel s.e., fantasies used + divergent discarded)"
    )
    for r, ci, oi, regret, hit, cells in out["rows"]:
        txt = "  ".join(f"L{lv}:{g:6.2f} ({se:.2f},{nf}+{nd}x)" for lv, g, se, nf, nd in cells)
        print(
            f"{r['seed']:3d} {r['p']:5.2f}   {ci}          {oi}            {regret:5.2f} "
            f"{'ok ' if hit else 'MISS'}   {out['excluded'][r['seed']]} | {txt}"
        )
    print(f"\nexcluded candidates (no valid fantasy) per truth: {out['excluded']}")
    print(f"truths with no candidate (counted as misses): {out['no_candidates']}")
    print(
        f"Criterion (i): regret >= 0.8 in {out['hits']} of {len(res)} truths (need {args.need} of {args.n})"
    )
    ok2 = True
    for lev in range(5):
        v = np.array([g for g in out["ratios"][lev] if np.isfinite(g)])
        med = float(np.median(v)) if v.size else float("nan")
        good = 0.5 <= med <= 2.0
        ok2 &= good
        print(
            f"Criterion (ii) level {lev}: median acq/oracle gain = {med:.2f} over {v.size} truths "
            f"-> {'ok' if good else 'FAIL'}"
        )
    cs = [x for r in res for x in r["candidates"]]
    rh = [x["rhat_p0"] for x in cs if np.isfinite(x["rhat_p0"])]
    nd = sum(x.get("n_discarded", 0) for x in cs)
    nr = sum(x.get("n_rhat_dropped", 0) for x in cs)
    nf = sum(x["n_fantasies"] + x.get("n_discarded", 0) for x in cs)
    print(f"rhat(log p0): max over refits {max(rh) if rh else float('nan'):.3f}")
    print(f"fantasies dropped for rhat(log p0) > 1.05: {nr}")
    print(f"divergent fantasies discarded: {nd} of {nf} fantasies ({100 * nd / max(nf, 1):.2f}%)")
    se = [x["rel_se"] for x in cs if np.isfinite(x["rel_se"])]
    nfs = [x["n_fantasies"] for x in cs]
    print(
        f"oracle rel s.e.: median {np.median(se) if se else float('nan'):.2f}, "
        f"max {max(se) if se else float('nan'):.2f}; fantasies per candidate: median {int(np.median(nfs))}"
    )
    ok = out["hits"] >= args.need and ok2 and len(res) >= args.n
    print("A12:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
