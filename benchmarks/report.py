"""PROTOCOL Section 5: figures, coverage table, false-success table and a markdown summary.

    uv run python -m benchmarks.report [--fast] [--out benchmarks/results/report]

Reads every result JSON under results/full (or results/fast). Writes, in --out:
  fig_<problem>.png      cost used / C* (x) against max |error| / eps (y), one marker per method and kappa
                         (median over seeds, bars from the 25th to the 75th percentile);
  coverage.csv, false_success.csv, summary.md.

Definitions (benchmarks/metrics.py): claimed = the method's own sigma_epi meets the tolerance; success =
claimed AND the truth is inside m_y +- 2 sigma_epi at >= 95% of Sigma_N; false claim = claimed and not
success. Coverage is pooled over runs and points of Sigma_N; its binomial limits (Clopper-Pearson, 95%)
treat points as independent, so they are narrower than the truth allows: points of one run are correlated.
The pass criteria of PROTOCOL Section 4 are evaluated for gcbml; "beats a baseline on a problem" is
operationalised as: higher success rate pooled over kappas, ties broken by lower mean cost used / C*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

from benchmarks import config  # noqa: E402
from benchmarks.certificate import RESULTS  # noqa: E402
from benchmarks.runner import results_root  # noqa: E402

METHOD_ORDER = ("gcbml", "a", "b", "c", "d", "e", "f")
COLORS = {
    "gcbml": "#1b1b1b",
    "a": "#1f77b4",
    "b": "#ff7f0e",
    "c": "#2ca02c",
    "d": "#d62728",
    "e": "#9467bd",
    "f": "#8c564b",
}
MARKERS = {1.0: "D", 1.5: "o", 2.0: "s", 4.0: "^"}


def load_results(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.rglob("seed*.json"))]


def binom_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def _group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def coverage_table(rows: list[dict]) -> list[dict]:
    """Pooled 95%-interval coverage per (problem, method): hits over (run, point) pairs."""
    out = []
    for (prob, meth), g in sorted(_group(rows, "problem", "method").items()):
        zs = [np.abs(np.asarray(r["z"])) for r in g if r.get("z")]
        if not zs:
            continue
        hits = int(sum((z <= 1.96).sum() for z in zs))
        n = int(sum(len(z) for z in zs))
        lo, hi = binom_ci(hits, n)
        out.append(
            {
                "problem": prob,
                "method": meth,
                "runs": len(zs),
                "points": n,
                "coverage": hits / n,
                "lo": lo,
                "hi": hi,
                "within": bool(lo <= 0.95 <= hi),
            }
        )
    return out


def false_success_table(rows: list[dict]) -> list[dict]:
    out = []
    for (prob, kap, meth), g in sorted(_group(rows, "problem", "kappa", "method").items()):
        n = len(g)
        claimed = sum(bool(r["claimed"]) for r in g)
        succ = sum(bool(r["success"]) for r in g)
        false = sum(bool(r["false_claim"]) for r in g)
        out.append(
            {
                "problem": prob,
                "kappa": kap,
                "method": meth,
                "runs": n,
                "claimed": claimed / n,
                "success": succ / n,
                "false_claim": false / n,
                "mean_cost_over_Cstar": float(np.mean([r["cost_over_Cstar"] for r in g])),
            }
        )
    return out


def _csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    path.write_text("\n".join([",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in rows]) + "\n")


def figure(rows: list[dict], problem: str, path: Path) -> None:
    g = _group(
        [r for r in rows if r["problem"] == problem and r.get("err_over_eps") is not None], "method", "kappa"
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for (meth, kap), rr in sorted(g.items()):
        x = np.array([r["cost_over_Cstar"] for r in rr])
        y = np.array([r["err_over_eps"] for r in rr])
        yq = np.percentile(y, [25, 50, 75])
        ax.errorbar(
            np.median(x),
            yq[1],
            yerr=[[yq[1] - yq[0]], [yq[2] - yq[1]]],
            fmt=MARKERS.get(kap, "o"),
            color=COLORS.get(meth, "gray"),
            ms=5,
            lw=0.8,
            capsize=2,
            label=None,
        )
    ax.axhline(1.0, color="0.7", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cost used / C*")
    ax.set_ylabel("max |error| / eps")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [
        plt.Line2D([], [], color=COLORS[m], marker="o", ls="", label=m)
        for m in METHOD_ORDER
        if any(k[0] == m for k in g)
    ]
    handles += [
        plt.Line2D([], [], color="0.4", marker=MARKERS[k], ls="", label=f"kappa {k:g}")
        for k in sorted({kk[1] for kk in g})
        if k in MARKERS
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pass_criteria(rows: list[dict]) -> list[str]:
    lines = []
    g = [r for r in rows if r["method"] == "gcbml"]
    feas = [p for p in config.FEASIBLE]
    # 1. success >= 80% at kappa = 2 on B1-B4 and B6 (B3 is not run: see config)
    for p in ("b1_poisson", "b2_upwind", "b3_mixing", "b4_sde", "b6_heat"):
        rr = [r for r in g if r["problem"] == p and r["kappa"] == 2.0]
        if not rr:
            lines.append(f"- success >= 80% at kappa = 2, {p}: not run")
            continue
        s = np.mean([r["success"] for r in rr])
        lines.append(
            f"- success >= 80% at kappa = 2, {p}: {s:.0%} of {len(rr)} runs: {'pass' if s >= 0.8 else 'FAIL'}"
        )
    # 2. pooled coverage
    cov = [c for c in coverage_table(g)]
    if cov:
        hits = sum(c["coverage"] * c["points"] for c in cov)
        n = sum(c["points"] for c in cov)
        lo, hi = binom_ci(int(round(hits)), n)
        lines.append(
            f"- pooled 95% coverage of gcbml: {hits / n:.3f} (limits {lo:.3f}-{hi:.3f}): "
            f"{'pass' if lo <= 0.95 <= hi else 'FAIL'}"
        )
    # 3. false success
    fs = [r for r in g if r["problem"] in (*feas, config.TRAP)]
    if fs:
        f = np.mean([r["false_claim"] for r in fs if r["problem"] != config.TRAP])
        lines.append(
            f"- false success (claimed but wrong) of gcbml on B1-B6: {f:.1%}: "
            f"{'pass' if f <= 0.05 else 'FAIL'}"
        )
    t7 = [r for r in g if r["problem"] == config.TRAP]
    if t7:
        c7 = np.mean([r["claimed"] for r in t7])
        lines.append(
            f"- claimed success of gcbml on B7: {c7:.1%} of {len(t7)} runs: "
            f"{'pass' if c7 <= 0.05 else 'FAIL'}"
        )
    # 4. beats each baseline on >= 4 of the 6 feasible problems
    wins = {}
    for m in METHOD_ORDER[1:]:
        w = 0
        for p in feas:
            a = [r for r in rows if r["problem"] == p and r["method"] == "gcbml"]
            b = [r for r in rows if r["problem"] == p and r["method"] == m]
            if not a or not b:
                continue
            sa, sb = np.mean([r["success"] for r in a]), np.mean([r["success"] for r in b])
            ca, cb = np.mean([r["cost_over_Cstar"] for r in a]), np.mean([r["cost_over_Cstar"] for r in b])
            w += int(sa > sb or (sa == sb and ca < cb))
        wins[m] = w
    for m, w in wins.items():
        lines.append(
            f"- gcbml beats {m} on {w} of {len(feas)} run feasible problems (needs 4 of 6): "
            f"{'pass' if w >= 4 else 'FAIL'}"
        )
    return lines


def write_report(rows: list[dict], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    problems = sorted({r["problem"] for r in rows})
    for p in problems:
        figure(rows, p, out / f"fig_{p}.png")
    cov, fs = coverage_table(rows), false_success_table(rows)
    _csv(cov, out / "coverage.csv")
    _csv(fs, out / "false_success.csv")
    md = [
        "# Benchmark summary",
        "",
        f"{len(rows)} runs, problems: {', '.join(problems)}",
        "",
        "## Coverage of the 95% intervals (pooled over runs and points of Sigma_N)",
        "",
        "| problem | method | runs | coverage | limits | within |",
        "|---|---|---|---|---|---|",
    ]
    md += [
        f"| {c['problem']} | {c['method']} | {c['runs']} | {c['coverage']:.3f} | "
        f"{c['lo']:.3f}-{c['hi']:.3f} | "
        f"{'yes' if c['within'] else 'no'} |"
        for c in cov
    ]
    md += [
        "",
        "## Claims and successes",
        "",
        "| problem | kappa | method | runs | claimed | success | false claim | cost / C* |",
        "|---|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {r['problem']} | {r['kappa']:g} | {r['method']} | {r['runs']} | {r['claimed']:.0%} | "
        f"{r['success']:.0%} | {r['false_claim']:.0%} | {r['mean_cost_over_Cstar']:.2f} |"
        for r in fs
    ]
    md += ["", "## Pass criteria for v1 (gcbml)", ""] + pass_criteria(rows)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return out / "summary.md"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = results_root(a.fast)
    rows = load_results(root)
    out = Path(a.out) if a.out else RESULTS / ("report_fast" if a.fast else "report")
    print(write_report(rows, out))


if __name__ == "__main__":
    main()
