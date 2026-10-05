"""Aggregate the A12 truths (spec Step 1): regret criterion (i) and per-level calibration (ii).

    uv run python -m benchmarks.a12.aggregate_a12 results/a12 [--n 20]

Criterion (i): per truth, the oracle value per cost of the acquisition's choice (the candidate with the best
acquisition ratio among the five per-level bests) is >= 0.8 x the best oracle value per cost; needed in at
least 16 of 20 truths. Criterion (ii): for every level, the median over truths of (acquisition gain / oracle
gain) lies in [0.5, 2]. Prints the table (per truth: choice level, oracle best level, regret ratio, gains per
level, s.e. and fantasies) and the verdict. Exit code 0 only if both criteria hold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--need", type=int, default=16)
    args = ap.parse_args(argv)
    files = sorted(Path(args.dir).glob("truth_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    res = [json.loads(f.read_text()) for f in files]
    print(f"{len(res)} of {args.n} truths found")
    hdr = "seed   p   choice_lvl oracle_best_lvl regret  | gain acq/oracle per level 0..4 (oracle rel s.e., fantasies used + divergent discarded)"
    print(hdr)
    hits, ratios = 0, {lev: [] for lev in range(5)}
    for r in res:
        c = r["candidates"]
        acq_i = int(np.argmax([x["acq_ratio"] for x in c]))
        orc_i = int(np.argmax([x["oracle_value"] for x in c]))
        best = c[orc_i]["oracle_value"]
        regret = c[acq_i]["oracle_value"] / best if best > 0 else float("nan")
        hit = bool(regret >= 0.8) if best > 0 else True
        hits += hit
        cells = []
        for x in c:
            g = x["acq_gain"] / x["oracle_gain"] if x["oracle_gain"] > 0 else float("nan")
            ratios[x["level"]].append(g)
            cells.append(
                f"L{x['level']}:{g:6.2f} ({x['rel_se']:.2f},{x['n_fantasies']}+{x.get('n_discarded', 0)}x)"
            )
        print(
            f"{r['seed']:3d} {r['p']:5.2f}   {c[acq_i]['level']}          {c[orc_i]['level']}"
            f"            {regret:5.2f} {'ok ' if hit else 'MISS'} | " + "  ".join(cells)
        )
    print(f"\nCriterion (i): regret >= 0.8 in {hits} of {len(res)} truths (need {args.need} of {args.n})")
    ok2 = True
    for lev in range(5):
        v = np.array([g for g in ratios[lev] if np.isfinite(g)])
        med = float(np.median(v)) if v.size else float("nan")
        good = 0.5 <= med <= 2.0
        ok2 &= good
        print(
            f"Criterion (ii) level {lev}: median acq/oracle gain = {med:.2f} over {v.size} truths "
            f"-> {'ok' if good else 'FAIL'}"
        )
    rh = [x["rhat_p0"] for r in res for x in r["candidates"] if np.isfinite(x["rhat_p0"])]
    print(f"rhat(log p0): max over refits {max(rh):.3f}; base {max(r['rhat_p0_base'] for r in res):.3f}")
    se = [x["rel_se"] for r in res for x in r["candidates"]]
    print(
        f"oracle rel s.e.: median {np.median(se):.2f}, max {max(se):.2f}; fantasies per candidate: "
        f"median {int(np.median([x['n_fantasies'] for r in res for x in r['candidates']]))}"
    )
    ok = hits >= args.need and ok2 and len(res) >= args.n
    print("A12:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys