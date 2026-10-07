# Pre-asymptotic benchmark

Principle under test: a lower-fidelity level must not hurt the converged-state estimate much.
On real CFD data, adding the under-resolved levels L6-L7 widened the h -> 0 band (15% vs 6.5%) and lowered the
order p (0.6 vs 1.4). This benchmark checks that effect on synthetic truths with a known answer. It was written
BEFORE any fix. It is expected to fail on the current model.

## Generator (`gcbml.synthetic.preasymptotic_truth`)

    z(x, hbar) = mu(x) - a(x) b(hbar)
    b(hbar)    = hbar^p / (1 + (hbar/h_s)^m)^(p/m)
    mu(x)      = mu0 + mu1 x^2,   a(x) = a0 (1 + a1 x^2),   x in [0, 1]

z is the log of the quantity of interest. The truth at h = 0 is z0(x) = mu(x). For hbar << h_s, b ~ hbar^p.
For hbar >> h_s, b -> h_s^p (flat). h_s = inf (the control family) gives b = hbar^p.
Observations: z_obs = z + N(0, s^2), one run per (x, level). hbar_L = 2^-(L - Lmin).

## Setting (assumptions in `config.py`)

Levels L6..L10 (hbar = 1 ... 1/16). 9 equispaced x in [0, 1]. s = 0.015. mu0 = log(0.002).
p ~ U[1, 2], h_s log-uniform [0.25, 0.6] (nominal [0.2, 0.6], narrowed to hit the L6->L7 band), m ~ U[2, 8],
mu1 ~ U[0.5, 1.5], a0 ~ U[1.4, 2.8], a1 ~ U[0, 1]. Control: same, h_s = inf, a0 ~ U[1.1, 3.4].

## Calibration (200 truths; median over x of |f_L - f_{L-1}| / f_L, then median over truths)

| pair | target (band) | achieved median | 10-90% across truths |
|---|---|---|---|
| L6->L7 | 8% (4-12%) | 4.4% | 0.2-21% |
| L7->L8 | 17% (12-22%) | 20.5% | 6.5-35% |
| L8->L9 | 18% (13-23%) | 15.9% | 9.7-24% |
| L9->L10 | 6% (2-10%) | 6.3% | 3.2-12% |

All four bands are met. Control family L8->L9: 17.5% (calibrated); its other rows are not calibrated.
Reproduce: `uv run python -m benchmarks.preasymptotic.calibrate`.

## Fits

Model A = levels L6-L10. Model O = levels L8-L10. Both: the current model, configuration copied from
`scripts/gcbml_hydro_fit_v3.py` (twy2 kernel, constant mean, increasing, log transform, S_mu 0.8, S_c 0.5,
S_delta 0.5, S_noise 0.02, log p ~ N(log 1.2, 0.45^2), 600 warm-up, 600 samples, 4 chains, default extensions).
A fit with max rhat > 1.05 after extension is dropped and re-run with a new key.
20 pre-asymptotic truths (seeds 0-19), 20 control truths (seeds 100-119).
Per truth and model: coverage of the 95% epistemic interval of f(x, 0) = exp(mu(x)) at the 9 x; relative width
W = mean_x (q975 - q025)/median; relative error of the median; posterior median of p.

## Pre-registered criteria (do not alter)

- C1: pooled coverage of model A on the pre-asymptotic family >= 0.90 (pooled over 20 truths x 9 x = 180 indicators).
- C2: median over truths of W_A / W_O on the pre-asymptotic family <= 1.25.
- C3: median over truths of W_A / W_O on the CONTROL family <= 1.00, and pooled coverage of model A on control >= 0.90.
- C4: pooled coverage of model O on the pre-asymptotic family >= 0.90 (sanity check that L8-L10 is itself fine here).

`tests/benchmarks/test_preasymptotic.py` (marked slow) asserts C1, C2, C3.

## Run (on a compute node)

    uv run python -m benchmarks.preasymptotic.run fit pre
    uv run python -m benchmarks.preasymptotic.run fit control
    uv run python -m benchmarks.preasymptotic.run report     # baseline.json, baseline.md, baseline.png
    uv run pytest -m slow tests/benchmarks/test_preasymptotic.py -n0

Fits resume from `results/fits_<family>.json`.
