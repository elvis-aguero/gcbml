"""aggregate_a12: a candidate without a valid fantasy is excluded, not a crash."""

import json

import numpy as np

from benchmarks.a12 import aggregate_a12 as ag


def cand(level, acq_ratio, value, n=10, gain=None, acq_gain=1.0):
    gain = value * 2.0 if gain is None else gain
    return dict(
        level=level, acq_ratio=acq_ratio, oracle_value=value, oracle_gain=gain, acq_gain=acq_gain, n_fantasies=n,
        n_discarded=0, n_rhat_dropped=0, rel_se=0.1 if n else float("nan"), rhat_p0=1.01, cost_mean=2.0,
    )  # fmt: skip


def truth(seed, cands):
    return dict(seed=seed, p=1.0, rhat_p0_base=float("nan"), candidates=cands)


def test_a_candidate_with_no_valid_fantasy_is_excluded_and_counted_without_a_crash(tmp_path, capsys):
    empty = cand(3, 0.1, float("nan"), n=0, gain=float("nan"))
    t0 = truth(0, [cand(0, 5.0, 4.0), cand(1, 3.0, 2.0), empty, cand(2, 1.0, 1.0), cand(4, 0.5, 0.2)])
    t1 = truth(1, [cand(0, 5.0, 4.0), cand(1, 3.0, 6.0), empty, cand(2, 1.0, 1.0), cand(4, 0.5, 0.2)])
    for t in (t0, t1):
        (tmp_path / f"truth_{t['seed']}.json").write_text(json.dumps(t))
    out = ag.aggregate([t0, t1], need=1, n=2)
    assert out["excluded"] == {0: 1, 1: 1}
    assert out["hits"] == 1  # truth 0: regret 1.0; truth 1: acquisition value 4 < 0.8 x 6
    assert np.all(np.isnan(out["ratios"][3]))  # the excluded level-3 candidate gives no ratio
    assert ag.main([str(tmp_path), "--n", "2", "--need", "1"]) in (0, 1)  # prints, does not raise
    assert "excluded" in capsys.readouterr().out


def test_an_acquisition_choice_without_a_valid_fantasy_is_a_miss():
    t = truth(0, [cand(0, 5.0, float("nan"), n=0, gain=float("nan")), cand(1, 3.0, 2.0)])
    out = ag.aggregate([t], need=1, n=1)
    assert out["hits"] == 0 and out["excluded"] == {0: 1}


def test_a_truth_without_any_candidate_is_a_counted_miss():
    t = truth(3, [])
    out = ag.aggregate([t], need=1, n=1)
    assert out["hits"] == 0 and out["excluded"] == {3: 0} and out["no_candidates"] == [3]
