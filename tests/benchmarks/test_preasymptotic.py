"""Pre-registered criteria of the pre-asymptotic benchmark (benchmarks/preasymptotic/README.md).

Reads fits_{pre,control}.json in benchmarks/preasymptotic/results (written by
`python -m benchmarks.preasymptotic.run fit pre|control`). Expected to FAIL on the current model (red phase).
"""

import pytest

from benchmarks.preasymptotic import run as R

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def crit():
    for w in ("pre", "control"):
        if not (R.RESULTS / f"fits_{w}.json").exists():
            pytest.skip(f"run `python -m benchmarks.preasymptotic.run fit {w}` first")
    return R.criteria(R.load_results())


def test_c1_coverage_of_all_levels_model(crit):
    assert crit["C1"] >= 0.90, (
        f"C1: pooled coverage of model A (L6-L10), pre-asymptotic family = {crit['C1']:.3f} < 0.90"
    )


def test_c2_width_ratio_pre_asymptotic(crit):
    assert crit["C2"] <= 1.25, f"C2: median W_A/W_O, pre-asymptotic family = {crit['C2']:.3f} > 1.25"


def test_c3_control_family(crit):
    assert crit["C3_ratio"] <= 1.00, f"C3: median W_A/W_O, control family = {crit['C3_ratio']:.3f} > 1.00"
    assert crit["C3_cov"] >= 0.90, (
        f"C3: pooled coverage of model A, control family = {crit['C3_cov']:.3f} < 0.90"
    )
