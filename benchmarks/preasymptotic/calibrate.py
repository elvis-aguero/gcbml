"""Calibration table: level-to-level relative change of f = exp(z) over 200 truths of a family."""

import numpy as np

from benchmarks.preasymptotic import config as C
from gcbml.synthetic import preasymptotic_truth


def level_changes(family: dict, seeds) -> np.ndarray:
    """(n_truths, 4): median over x of |f_L - f_{L-1}| / f_L for the four level pairs."""
    out = []
    for s in seeds:
        t = preasymptotic_truth(int(s), **family)
        f = np.array([np.exp(t.z(C.X_GRID, h)) for h in C.HBAR])
        out.append(np.median(np.abs(f[1:] - f[:-1]) / f[1:], axis=1))
    return np.array(out)


def table(family: dict, seeds=range(1000, 1000 + C.N_CALIB)) -> tuple[str, bool]:
    ch = level_changes(family, seeds)
    lines = ["pair      target  band          median  p10-p90        in band"]
    ok = True
    for j, (name, (tgt, (lo, hi))) in enumerate(C.TARGETS.items()):
        med = float(np.median(ch[:, j]))
        p10, p90 = np.percentile(ch[:, j], [10, 90])
        inb = lo <= med <= hi
        ok &= inb
        lines.append(f"{name:8s}  {tgt:5.2f}  {lo:.2f}-{hi:.2f}   {med:6.3f}  {p10:.3f}-{p90:.3f}   {inb}")
    return "\n".join(lines), ok


if __name__ == "__main__":
    for name, fam in (("pre-asymptotic", C.FAMILY), ("control", C.CONTROL)):
        txt, ok = table(fam)
        note = "" if name == "pre-asymptotic" else " (control: only the L8->L9 row is calibrated)"
        print(f"== {name} family ==\n{txt}\nall four bands met: {ok}{note}\n")
