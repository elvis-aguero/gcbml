"""Runner and submit: result files, resume, manifest and sbatch script."""

import json

import pytest
from toy_problem import toy_setup

from benchmarks import runner, submit
from benchmarks.runner import result_path, results_root, run_task

C_STAR = 48.0


def write_cert(root, fast=True):
    p = root / "certificates" / f"toy{'_fast' if fast else ''}.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "problem": "toy",
                "rel_tol": 0.5,
                "fast": fast,
                "certificate": {"C_star": C_STAR},
                "evaluated": [],
            }
        )
    )


def test_task_writes_one_json_with_the_protocol_fields_and_resumes(tmp_path, monkeypatch):
    write_cert(tmp_path)
    setup = toy_setup(rel_tol=0.5)
    path = run_task("toy", 4.0, "a", 1, fast=True, root=tmp_path, setup=setup)
    assert path == result_path(results_root(True, tmp_path), "toy", 4.0, "a", 1)
    rec = json.loads(path.read_text())
    for k in (
        "success",
        "claimed",
        "false_claim",
        "cost_over_Cstar",
        "err_over_eps",
        "coverage95",
        "z",
        "gates",
        "allocation",
        "wall_seconds",
        "budget",
        "C_star",
        "method",
        "seed",
        "kappa",
    ):
        assert k in rec, k
    assert rec["budget"] == 4.0 * C_STAR and rec["cost_over_Cstar"] <= 4.0 and len(rec["z"]) == 200
    # resume: a second call must not run the method again
    monkeypatch.setattr("benchmarks.baselines.run_method", lambda *a, **k: pytest.fail("recomputed"))
    assert run_task("toy", 4.0, "a", 1, fast=True, root=tmp_path, setup=setup) == path


def test_failure_leaves_an_error_file_and_no_result(tmp_path, monkeypatch):
    write_cert(tmp_path)
    monkeypatch.setattr("benchmarks.baselines.run_method", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        run_task("toy", 2.0, "a", 3, fast=True, root=tmp_path, setup=toy_setup())
    p = result_path(results_root(True, tmp_path), "toy", 2.0, "a", 3)
    assert not p.exists() and p.with_suffix(".error.txt").exists()


def test_manifest_counts_and_sbatch_script(tmp_path):
    full = submit.all_tasks("full")
    assert len(full) == (5 * 3 + 1) * 7 * 20
    assert {t["problem"] for t in full} == {
        "b1_poisson",
        "b2_upwind",
        "b4_sde",
        "b5_kink",
        "b6_heat",
        "b7_thin_layer",
    }
    assert len({(t["problem"], t["kappa"], t["method"], t["seed"]) for t in full}) == len(full)
    smoke = submit.all_tasks("smoke")
    assert {(t["problem"], t["kappa"], t["seed"]) for t in smoke} == {("b1_poisson", 1.5, 1)}
    assert len(smoke) == 7
    m = submit.write_manifest("smoke", tmp_path)
    assert len(json.loads(m.read_text())["tasks"]) == 7
    # a finished task drops out of the next manifest
    p = result_path(results_root(True, tmp_path), "b1_poisson", 1.5, "a", 1)
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    assert len(json.loads(submit.write_manifest("smoke", tmp_path).read_text())["tasks"]) == 6
    s = submit.sbatch_script(m, 6, "01:00:00", "8G", "x")
    assert "--array=0-5%100" in s and "--cpus-per-task=4" in s and "--partition=batch" in s
    assert "--task-id $SLURM_ARRAY_TASK_ID" in s


@pytest.mark.slow
def test_smoke_end_to_end_with_gcbml_on_the_toy(tmp_path):
    write_cert(tmp_path)
    path = run_task("toy", 2.0, "gcbml", 1, fast=True, root=tmp_path, setup=toy_setup(0.5))
    rec = json.loads(path.read_text())
    assert rec["cost_used"] <= rec["budget"] * (1 + 1e-9) and rec["status"] in {
        "success",
        "P2",
        "uncalibrated",
    }


def test_cli_reads_a_manifest(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(runner, "run_task", lambda *a, **k: called.update(a=a) or tmp_path)
    monkeypatch.setattr(runner, "enable_jax_cache", lambda: None)
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            {
                "fast": True,
                "tasks": [
                    {"problem": "p", "kappa": 2.0, "method": "a", "seed": 4},
                    {"problem": "q", "kappa": 4.0, "method": "c", "seed": 9},
                ],
            }
        )
    )
    runner.main(["--manifest", str(m), "--task-id", "1"])
    assert called["a"] == ("q", 4.0, "c", 9, True)
