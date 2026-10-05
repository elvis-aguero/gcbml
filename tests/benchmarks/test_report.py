"""Report on fake results with hand-computable tables."""

import json

import numpy as np
import pytest

from benchmarks import report
from benchmarks.runner import result_path, results_root


def fake(problem, kappa, method, seed, claimed, success, z, cost=1.0, err=0.5):
    return {
        "problem": problem,
        "kappa": kappa,
        "method": method,
        "seed": seed,
        "claimed": claimed,
        "success": success,
        "false_claim": claimed and not success,
        "z": z,
        "cost_over_Cstar": cost,
        "err_over_eps": err,
        "coverage95": float(np.mean(np.abs(z) <= 1.96)),
    }


@pytest.fixture
def results(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for problem in ("b1_poisson", "b7_thin_layer"):
        for kappa in (1.5, 2.0):
            for method in ("gcbml", "a"):
                for seed in range(1, 5):
                    z = list(rng.standard_normal(10))
                    good = method == "gcbml"
                    rows.append(
                        fake(
                            problem,
                            kappa,
                            method,
                            seed,
                            claimed=good or seed == 1,
                            success=good,
                            z=z,
                            cost=1.0 + 0.1 * seed,
                            err=0.5 + seed,
                        )
                    )
    for r in rows:
        p = result_path(results_root(True, tmp_path), r["problem"], r["kappa"], r["method"], r["seed"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r))
    return tmp_path, rows


def test_load_and_tables_match_hand_counts(results):
    root, rows = results
    loaded = report.load_results(results_root(True, root))
    assert len(loaded) == len(rows) == 32
    fs = {(r["problem"], r["kappa"], r["method"]): r for r in report.false_success_table(loaded)}
    a = fs[("b1_poisson", 2.0, "a")]  # claimed only at seed 1, never successful: 1 of 4 false claims
    assert (a["runs"], a["claimed"], a["success"], a["false_claim"]) == (4, 0.25, 0.0, 0.25)
    g = fs[("b1_poisson", 2.0, "gcbml")]
    assert (g["claimed"], g["success"], g["false_claim"]) == (1.0, 1.0, 0.0)
    assert g["mean_cost_over_Cstar"] == pytest.approx(np.mean([1.1, 1.2, 1.3, 1.4]))
    cov = {(c["problem"], c["method"]): c for c in report.coverage_table(loaded)}
    zs = np.abs(
        np.concatenate([r["z"] for r in loaded if r["problem"] == "b1_poisson" and r["method"] == "a"])
    )
    c = cov[("b1_poisson", "a")]
    assert c["points"] == len(zs) == 80 and c["coverage"] == pytest.approx(np.mean(zs <= 1.96))
    assert c["lo"] <= c["coverage"] <= c["hi"]


def test_binomial_limits_against_the_known_interval():
    # Clopper-Pearson for 5 of 10: (0.1871, 0.8129)
    lo, hi = report.binom_ci(5, 10)
    assert lo == pytest.approx(0.18709, abs=1e-4) and hi == pytest.approx(0.81291, abs=1e-4)
    assert report.binom_ci(0, 10)[0] == 0.0 and report.binom_ci(10, 10)[1] == 1.0


def test_report_writes_figures_tables_and_summary(results, tmp_path):
    root, _ = results
    out = tmp_path / "out"
    md = report.write_report(report.load_results(results_root(True, root)), out)
    assert (out / "fig_b1_poisson.png").stat().st_size > 1000 and (out / "fig_b7_thin_layer.png").exists()
    assert (out / "coverage.csv").read_text().startswith("problem,method,runs")
    text = md.read_text()
    assert "Coverage" in text and "false claim" in text and "Pass criteria" in text
    # gcbml claims success on B7 in every run of this fake set: the criterion must fail
    assert "claimed success of gcbml on B7: 100.0% of 8 runs: FAIL" in text


def test_figure_is_minimal_and_has_its_legend_outside_the_axes(results, tmp_path):
    import matplotlib.pyplot as plt

    root, _ = results
    rows = report.load_results(results_root(True, root))
    captured = {}
    orig = plt.close

    def spy(fig=None):
        fig.canvas.draw()
        ax = fig.axes[0]
        captured.update(
            ax=ax, legend_x0=ax.get_legend().get_window_extent().x0, axes_x1=ax.get_window_extent().x1
        )
        orig(fig)

    plt.close = spy
    try:
        report.figure(rows, "b1_poisson", tmp_path / "f.png")
    finally:
        plt.close = orig
    ax = captured["ax"]
    assert ax.get_title() == "" and "cost" in ax.get_xlabel() and "error" in ax.get_ylabel()
    assert captured["legend_x0"] >= captured["axes_x1"] - 1e-6  # the legend sits outside the plotted area
