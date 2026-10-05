"""Profile one Campaign.ask() and one report() on an A12 problem (cold, then warm).

    PYTHONPATH=. uv run python scripts/profile_campaign.py --n0 8 --tag n45 [--extra3 0] [--warmup W --samples S]

Wraps the functions of campaign, inference, stacking, gates, acquisition, forecast and cost with a timer
that keeps a call stack, so every call gets an inclusive and an exclusive time and a path
("_gates>_fit_all>fit"). Nothing in campaign.py is edited: the wrappers replace attributes at run time.
Phases: cold ask (compiles), report (analysis cached by ask), then the same on a deep copy of the
campaign (warm: compiled programs reused). Prints the table (seconds, % of the phase) per phase.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import time
from collections import defaultdict

import jax

import gcbml  # noqa: F401
from benchmarks.a12.run_truth import COST_PRIOR, SCALES
from gcbml import acquisition as acq
from gcbml import campaign as cmp
from gcbml import cost, forecast, gates, inference, stacking
from gcbml.data import Probe
from gcbml.synthetic import A12Truth

STACK: list[str] = []
EVENTS: list[tuple[str, float, float]] = []  # (path, inclusive, exclusive)
CHILD: list[float] = [0.0]


def timed(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        STACK.append(name)
        CHILD.append(0.0)
        t0 = time.perf_counter()
        try:
            out = fn(*a, **k)
            try:
                jax.block_until_ready(out)
            except Exception:  # not an array pytree
                pass
        finally:
            dt = time.perf_counter() - t0
            kids = CHILD.pop()
            path = ">".join(STACK)
            STACK.pop()
            EVENTS.append((path, dt, dt - kids))
            CHILD[-1] += dt
        return out

    return w


def install():
    C = cmp.Campaign
    for n in (
        "_fit_all _stacking _moments _pool _analyse_once _gates _g5 _g7 _build_report _sigma_fid "
        "_cost_posterior _candidates _price _select _pending_candidates _repair_probe _cost_data"
    ).split():
        setattr(C, n, timed(n, getattr(C, n)))
    C._pit_components = staticmethod(timed("_pit_components", C._pit_components))
    cmp.inference.fit = timed("fit", inference.fit)
    for mod, names in (
        (stacking, "log_scores stack_weights cross_fitted_weights holdout"),
        (
            gates,
            "g0_order g1_level_holdout g2_block_loo g3_noise g4_pre_asymptotic g5_monotone g6_shape g7_prior",
        ),
        (acq, "select_batch sigma_epi_physical"),
        (forecast, "forecast"),
        (cost, "fit_cost predict_log2 cost_cap expected_cost"),
    ):
        for n in names.split():
            setattr(mod, n, timed(n, getattr(mod, n)))


def phase(label, fn):
    EVENTS.clear()
    t0 = time.perf_counter()
    fn()
    total = time.perf_counter() - t0
    excl = defaultdict(float)
    incl = defaultdict(float)
    cnt = defaultdict(int)
    for p, i, e in EVENTS:
        excl[p] += e
        incl[p] += i
        cnt[p] += 1
    rows = sorted(excl.items(), key=lambda kv: -kv[1])
    acc = sum(excl.values())
    print(f"\n== {label}: total {total:.1f} s ==")
    print(f"{'path':70s} {'calls':>5s} {'incl s':>8s} {'excl s':>8s} {'excl %':>7s}")
    for p, e in rows:
        if e / total < 0.002:
            continue
        print(f"{p:70s} {cnt[p]:5d} {incl[p]:8.1f} {e:8.1f} {100 * e / total:6.1f}%")
    print(
        f"{'(untimed remainder)':70s} {'':5s} {'':8s} {total - acc:8.1f} {100 * (total - acc) / total:6.1f}%"
    )
    return dict(total=total, excl=dict(excl), incl=dict(incl), calls=dict(cnt))


def build(args):
    t = A12Truth(args.seed, d=2, budget=1e9, eps_abs=args.eps)
    s = cmp.CampaignSettings(
        n_warmup=args.warmup, n_samples=args.samples, n_chains=4, n0_per_control=args.n0,
        n_candidates_u=args.n_u, seed=args.seed + 1000,
    )  # fmt: skip
    c = cmp.Campaign(t.problem(), SCALES, COST_PRIOR, s)
    probes = c.initial_design()
    if args.extra3:
        base = [p for p in probes if c._level_of(p.h) == 0][: args.extra3]
        probes = probes + [Probe(c._new_id("x"), p.u, c._h_of_level(3), None) for p in base]
    o = t.oracle(seed=args.seed)
    o.submit(probes, [1e18] * len(probes))
    c._pending.clear()
    c.tell(o.poll())
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n0", type=int, default=8)
    ap.add_argument("--extra3", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--n-u", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no-warm", action="store_true")
    args = ap.parse_args()
    install()
    c = build(args)
    print(f"rows n = {c._n_rows()}, settings {c.settings}", flush=True)
    c_warm = copy.deepcopy(c)
    out = {}
    out["cold_ask"] = phase("cold ask()", lambda: c.ask())
    out["cold_report"] = phase("cold report() after ask", lambda: c.report())
    if not args.no_warm:
        out["warm_ask"] = phase("warm ask()", lambda: c_warm.ask())
        out["warm_report"] = phase("warm report() after ask", lambda: c_warm.report())
    json.dump(out, open(f"/oscar/scratch/eaguerov/gcbml_profile_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
