"""Gates G0-G7 (spec Section 3, Step 3): a pass case, a fail case and a not-testable case for each.

Inputs are hand-computable or synthetic draws with known answers; no MCMC.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from test_acquisition import CFG, CFG_FLAT, make_data, make_params, simulate_z, stack

import gcbml  # noqa: F401
from gcbml import gates, model, transforms

# ----------------------------------------------------------------------------------------------
# G0
# ----------------------------------------------------------------------------------------------


def test_g0_clear_order_passes():
    rng = np.random.default_rng(0)
    log_p0 = np.log(2.0) + 0.2 * rng.standard_normal((4000, 2))  # P(p > 0.5) ~ 1, sd 0.2 <= 0.5
    r = gates.g0_order(log_p0)
    assert r.name == "G0" and r.status == "pass"
    assert np.all(np.asarray(r.stats["p_gt_half"]) >= 0.9) and np.all(np.asarray(r.stats["sd_log_p"]) < 0.25)


def test_g0_wide_order_fails_by_the_sd_rule_although_the_probability_rule_holds():
    rng = np.random.default_rng(1)
    log_p0 = 1.5 + 1.0 * rng.standard_normal((20000, 1))  # P(p > 0.5) = Phi(2.19) = 0.986, sd 1
    r = gates.g0_order(log_p0)
    assert float(np.asarray(r.stats["p_gt_half"])[0]) > 0.9
    assert r.status == "fail"


def test_g0_prior_like_draws_fail():
    # the prior log p ~ N(0, 1) gives P(p > 0.5) = 0.76 and sd 1: it must not pass G0 alone
    rng = np.random.default_rng(2)
    r = gates.g0_order(rng.standard_normal((20000, 1)), prior_sd=1.0)
    assert float(np.asarray(r.stats["p_gt_half"])[0]) == pytest.approx(0.756, abs=0.01)
    assert r.status == "fail"


def test_g0_sd_rule_scales_with_the_prior_sd():
    rng = np.random.default_rng(3)
    log_p0 = np.log(2.0) + 0.4 * rng.standard_normal((4000, 1))
    assert gates.g0_order(log_p0, prior_sd=1.0).status == "pass"  # 0.4 <= 0.5
    assert gates.g0_order(log_p0, prior_sd=0.6).status == "fail"  # 0.4 > 0.3


def test_g0_every_component_must_pass():
    rng = np.random.default_rng(4)
    good = np.log(2.0) + 0.2 * rng.standard_normal((4000, 1))
    bad = rng.standard_normal((4000, 1))
    assert gates.g0_order(np.hstack([good, bad])).status == "fail"


def test_g0_varying_order_needs_p_at_every_x():
    rng = np.random.default_rng(5)
    S, s = 2000, 3
    log_p0 = np.log(2.0) + 0.2 * rng.standard_normal((S, 1))
    sigma_pi = np.full((S, 1), 0.3)  # P(sigma_pi > 0.05) = 1: the order varies
    p_ok = 2.0 + 0.1 * rng.standard_normal((S, s, 1))
    assert gates.g0_order(log_p0, p_ok, sigma_pi).status == "pass"
    p_bad = p_ok.copy()
    p_bad[:, 1, 0] = 0.3 + 0.1 * rng.standard_normal(S)  # p < 0.5 at the second x
    r = gates.g0_order(log_p0, p_bad, sigma_pi)
    assert r.status == "fail"
    # the same bad p(x) is ignored when the order does not vary
    assert gates.g0_order(log_p0, p_bad, np.full((S, 1), 0.01)).status == "pass"


def test_g0_varying_order_without_p_of_x_is_not_testable():
    rng = np.random.default_rng(6)
    log_p0 = np.log(2.0) + 0.2 * rng.standard_normal((2000, 1))
    r = gates.g0_order(log_p0, None, np.full((2000, 1), 0.3))
    assert r.status == "not testable"


# ----------------------------------------------------------------------------------------------
# G1
# ----------------------------------------------------------------------------------------------


def exact_normal(n, seed, loc=0.0, scale=1.0):
    """n draws of N(loc, scale^2) shifted and rescaled to the exact sample mean loc and sample sd scale."""
    z = np.random.default_rng(seed).standard_normal(n)
    return loc + scale * (z - z.mean()) / z.std()


def test_g1_pass_for_standard_normal_scores():
    z = np.random.default_rng(16).standard_normal(400)
    r = gates.g1_level_holdout(z, np.abs(z) < 1.96)
    assert r.name == "G1" and r.status == "pass", r.stats
    assert r.stats["c_loo"] == pytest.approx(np.mean(z**2), rel=1e-12)  # Bachoc eq 6 with sigma^2 c^2 = 1


def test_g1_fails_coverage_for_scores_with_twice_the_sd():
    # the model claims unit scores; the truth has sd 2, so 95% intervals cover ~ 67%
    z = 2.0 * np.random.default_rng(11).standard_normal(400)
    r = gates.g1_level_holdout(z, np.abs(z) < 1.96)
    assert r.status == "fail"
    assert r.stats["coverage_pvalue"] < 0.05 and r.stats["c_loo"] > 2.0


def test_g1_fails_sign_bias_for_shifted_scores():
    z = 1.0 + np.random.default_rng(12).standard_normal(400)
    r = gates.g1_level_holdout(z, np.abs(z) < 1.96)
    assert r.status == "fail"
    assert abs(r.stats["mean_z"]) > 2 * r.stats["se_mean_z"]


@pytest.mark.parametrize("k", [84, 87, 89, 90, 92, 100])
def test_g1_coverage_limits_are_the_exact_binomial_limits(k):
    n = 100
    z = exact_normal(n, 13)  # mean 0 and mean z^2 = 1 exactly: only coverage can fail
    inside = np.arange(n) < k
    r = gates.g1_level_holdout(z, inside)
    p = stats.binomtest(k, n, 0.95).pvalue
    assert r.stats["coverage_pvalue"] == pytest.approx(p, rel=1e-9)
    assert (r.status == "pass") == (p >= 0.05)


def test_g1_c_loo_band():
    n = 200
    z0 = exact_normal(n, 14)
    band = 2.0 * np.sqrt(2.0 / n)
    r_in = gates.g1_level_holdout(z0 * np.sqrt(1.0 + 0.5 * band), np.ones(n, bool) & (np.arange(n) >= 5))
    # coverage 195/200 is fine; C_LOO = 1 + 0.5 band is inside the band
    assert r_in.stats["c_loo"] == pytest.approx(1.0 + 0.5 * band, rel=1e-9)
    assert r_in.status == "pass"
    r_out = gates.g1_level_holdout(z0 * np.sqrt(1.0 + 2.0 * band), np.arange(n) >= 5)
    assert r_out.stats["c_loo"] == pytest.approx(1.0 + 2.0 * band, rel=1e-9)
    assert r_out.status == "fail"


def test_g1_type_one_error_is_small_for_calibrated_scores():
    fails = 0
    for seed in range(100):
        z = np.random.default_rng(1000 + seed).standard_normal(150)
        fails += gates.g1_level_holdout(z, np.abs(z) < 1.96).status != "pass"
    assert fails <= 25  # three tests at 5% each: about 14% expected


def test_g1_not_testable():
    z = np.random.default_rng(15).standard_normal(50)
    assert gates.g1_level_holdout(z, np.abs(z) < 1.96, testable=False).status == "not testable"
    assert gates.g1_level_holdout(np.zeros(0), np.zeros(0, bool)).status == "not testable"


# ----------------------------------------------------------------------------------------------
# G2: block LOO
# ----------------------------------------------------------------------------------------------

NX, N_PAD = 14, 40


def loo_world(seed=0, nx=NX, within=False):
    xs = np.linspace(0.05, 0.95, nx)
    X = np.concatenate([xs[:, None]] * 2)
    H = np.concatenate([np.full((nx, 1), h) for h in (1.0, 0.5)])
    data = make_data(X, H, N_PAD)
    return data


def brute_force_block(params, data, z, cfg, block):
    """Refit with the block removed (fixed parameters): the predictive of the block from the other rows."""
    keep = np.asarray(data.mask).copy()
    keep[block] = False
    reduced = dataclasses.replace(data, mask=keep)
    jn = model.joint_new(
        params,
        reduced,
        z,
        cfg,
        np.asarray(data.X)[block],
        np.asarray(data.H)[block],
        np.asarray(params.P)[block],
        same_run=False,
        noise_var_new=np.asarray(params.noise_var)[block],
    )
    return np.asarray(jn.mean), np.asarray(jn.cov)


@pytest.mark.parametrize("cfg", [CFG, CFG_FLAT], ids=["gaussian_beta", "flat_beta"])
@pytest.mark.parametrize("block", [[3], [2, 9, 20], [0, 1, 14, 15]])
def test_block_loo_equals_brute_force_refits(cfg, block):
    data = loo_world()
    params = make_params(N_PAD, nv=2e-2)
    z = simulate_z(0, params, data, CFG)
    mean, cov = gates.block_loo(params, data, z, cfg, np.asarray(block))
    m_ref, c_ref = brute_force_block(params, data, z, cfg, block)
    assert np.allclose(mean, m_ref, atol=1e-8, rtol=0)
    assert np.allclose(cov, c_ref, atol=1e-8, rtol=0)


def test_block_loo_single_row_is_the_virtual_loo_formula():
    # Bachoc 1301.4320 Prop 3.1: y_i - yhat_i = (Q y)_i / Q_ii and c^2 = 1 / Q_ii, Q the precision of z
    data = loo_world()
    params = make_params(N_PAD, nv=2e-2)
    z = simulate_z(1, params, data, CFG)
    S = model._setup(params, data, CFG)
    real = np.asarray(data.mask)
    b0, bsd = (np.asarray(a) for a in CFG.beta_prior)
    Kt = np.asarray(S.K + (S.A * bsd**2) @ S.A.T)[np.ix_(real, real)]
    Q = np.linalg.inv(Kt)  # reference only
    r = (np.asarray(z) - np.asarray(S.rho0) - np.asarray(S.A @ b0))[real]
    i = 5
    mean, cov = gates.block_loo(params, data, z, CFG, np.asarray([i]))
    assert float(mean[0]) == pytest.approx(float(np.asarray(z)[i] - (Q @ r)[i] / Q[i, i]), abs=1e-8)
    assert float(cov[0, 0]) == pytest.approx(float(1.0 / Q[i, i]), rel=1e-8)


def test_block_loo_with_within_run_noise_equals_brute_force_for_a_whole_run():
    nx = 6
    xs = np.linspace(0.05, 0.95, nx)
    v = np.linspace(0.1, 0.9, 4)
    # d = 2: (u, v); three runs of 4 outputs each; the left-out block is the whole second run
    X = np.array([[u, vv] for u in xs[:3] for vv in v])
    H = np.full((len(X), 1), 0.5)
    n_pad = 16
    data = make_data(X, H, n_pad)
    data = dataclasses.replace(
        data, run=np.concatenate([np.repeat(np.arange(3), 4), np.full(n_pad - 12, -1)])
    )
    cfg = dataclasses.replace(CFG, within_run=True, v_index=(1,))
    params = make_params(n_pad, nv=2e-2)
    params = params._replace(
        ell_mu=jnp.asarray([0.3, 0.3]), delta=params.delta._replace(ell_x=jnp.asarray([[0.4, 0.4]]))
    )
    z = jnp.asarray(np.random.default_rng(0).normal(size=n_pad))
    block = np.arange(4, 8)
    mean, cov = gates.block_loo(params, data, z, cfg, block)
    keep = np.asarray(data.mask).copy()
    keep[block] = False
    jn = model.joint_new(
        params,
        dataclasses.replace(data, mask=keep),
        z,
        cfg,
        X[block],
        np.asarray(H)[block],
        np.asarray(params.P)[block],
        same_run=True,
        noise_var_new=np.asarray(params.noise_var)[block],
    )
    assert np.allclose(mean, jn.mean, atol=1e-8) and np.allclose(cov, jn.cov, atol=1e-8)


def test_u_statistic_is_the_determinant_of_eq_10():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(7, 3))
    assert gates.u_statistic(E) == pytest.approx(1.0 / np.linalg.det(np.eye(3) + E.T @ E), rel=1e-12)
    e = rng.normal(size=(5, 1))
    assert gates.u_statistic(e) == pytest.approx(
        1.0 / (1.0 + (e.T @ e).item()), rel=1e-12
    )  # k = 1 reduces to this


@pytest.mark.parametrize("k", [1, 2])
def test_u_statistic_has_the_beta_product_mean_under_the_matrix_t_reference(k):
    # Overstall & Woods 1506.04489 Section 3.1.2: for E ~ MT(0, I_k, I_n0, delta),
    # U ~ prod_s Beta((k + delta - s) / 2, n0 / 2), so E U = prod_s (k + delta - s) / (k + delta - s + n0).
    # Draw E = Z W^{-1/2}, W ~ Wishart_k(delta + k - 1, I).
    rng = np.random.default_rng(100 + k)
    n0, delta, reps = 6, 9, 20000
    us = np.empty(reps)
    for i in range(reps):
        Z = rng.standard_normal((n0, k))
        G = rng.standard_normal((delta + k - 1, k))
        W = G.T @ G
        L = np.linalg.cholesky(W)
        E = np.linalg.solve(L, Z.T).T  # E E^T-structure: E^T E = L^{-1} Z^T Z L^{-T}
        us[i] = gates.u_statistic(E)
    ref = np.prod([(k + delta - s) / (k + delta - s + n0) for s in range(1, k + 1)])
    assert us.mean() == pytest.approx(ref, abs=4 * us.std() / np.sqrt(reps))


def g2_setup(seed, truth_noise=0.0, nx=30):
    n_pad = 64
    xs = np.linspace(0.05, 0.95, nx)
    X = np.concatenate([xs[:, None]] * 2)
    H = np.concatenate([np.full((nx, 1), h) for h in (1.0, 0.5)])
    data = make_data(X, H, n_pad)
    plist = [make_params(n_pad, nv=1e-2, p=1.4 + 0.05 * s) for s in range(3)]
    z = np.asarray(simulate_z(seed, plist[0], data, CFG))
    z = z + truth_noise * np.random.default_rng(seed + 1).standard_normal(n_pad)
    blocks = [np.array([3 * i, 3 * i + 1, 3 * i + 2]) for i in range(20)]  # 20 blocks of 3 rows
    return plist, data, z, blocks


def test_g2_passes_when_the_model_is_the_truth():
    plist, data, z, blocks = g2_setup(0)
    same = [plist[0]] * 3
    r = gates.g2_block_loo(stack(same), data, jnp.stack([jnp.asarray(z)] * 3), CFG, blocks)
    assert r.name == "G2" and r.status == "pass", r.stats
    assert r.stats["n_outputs"] == 60
    # whitened block residuals are N(0, 1): the sum of the Mahalanobis distances is chi-square(60)
    assert stats.chi2.sf(r.stats["mahalanobis_total"], 60) > 0.01


def test_g2_fails_when_the_data_carry_unmodelled_noise():
    plist, data, z, blocks = g2_setup(0, truth_noise=0.3)  # the model claims noise variance 1e-2 (sd 0.1)
    r = gates.g2_block_loo(stack([plist[0]] * 3), data, jnp.stack([jnp.asarray(z)] * 3), CFG, blocks)
    assert r.status == "fail"


def test_g2_pools_draws_with_the_between_draw_spread():
    # two draws that disagree about the block: the pooled covariance must exceed each draw's own
    plist, data, z, blocks = g2_setup(0)
    p_a, p_b = plist[0], plist[0]._replace(c0=plist[0].c0 + 0.3)
    zz = jnp.stack([jnp.asarray(z)] * 2)
    r = gates.g2_block_loo(stack([p_a, p_b]), data, zz, CFG, blocks[:2], min_outputs=1)
    m_a, c_a = gates.block_loo(p_a, data, jnp.asarray(z), CFG, blocks[0])
    m_b, c_b = gates.block_loo(p_b, data, jnp.asarray(z), CFG, blocks[0])
    d = np.asarray(m_a - m_b)
    c_pool = 0.5 * (np.asarray(c_a) + np.asarray(c_b)) + 0.25 * np.outer(d, d)
    zb = np.asarray(z)[blocks[0]]
    mbar = 0.5 * np.asarray(m_a + m_b)
    L = np.linalg.cholesky(c_pool)
    e = np.linalg.solve(L, zb - mbar)
    assert r.stats["mahalanobis"][0] == pytest.approx(float(e @ e), rel=1e-8)


def test_g2_not_testable_with_too_few_outputs():
    plist, data, z, blocks = g2_setup(0)
    r = gates.g2_block_loo(stack([plist[0]]), data, jnp.asarray(z)[None], CFG, blocks[:2])
    assert r.status == "not testable"  # 6 left-out outputs < 10
    assert gates.g2_block_loo(stack([plist[0]]), data, jnp.asarray(z)[None], CFG, []).status == "not testable"


# ----------------------------------------------------------------------------------------------
# G3
# ----------------------------------------------------------------------------------------------


def test_g3_hand_case_matches_the_chi_square_p_value():
    # one group, two replicates 0 and 1: sum of squares about the mean 0.5, modelled s^2 = 0.5 -> T = 1, dof 1
    r = gates.g3_noise([np.array([0.0, 1.0])], [0.5])
    p = 2 * min(stats.chi2.cdf(1.0, 1), stats.chi2.sf(1.0, 1))
    assert r.stats["statistic"] == pytest.approx(1.0) and r.stats["dof"] == 1
    assert r.stats["p"] == pytest.approx(p, rel=1e-12)
    assert r.status == "pass"


def test_g3_pass_with_the_right_s2_and_fail_with_a_wrong_one():
    rng = np.random.default_rng(20)
    s2 = 0.09
    groups = [np.sqrt(s2) * rng.standard_normal(4) + rng.normal() for _ in range(25)]
    assert gates.g3_noise(groups, [s2] * 25).status == "pass"
    assert gates.g3_noise(groups, [s2 / 4] * 25).status == "fail"  # noise underestimated, T ~ 4 x dof
    assert gates.g3_noise(groups, [s2 * 4] * 25).status == "fail"  # overestimated: T ~ dof / 4


def test_g3_size_over_many_simulations():
    rng = np.random.default_rng(21)
    fails = 0
    for _ in range(200):
        groups = [rng.standard_normal(3) for _ in range(10)]
        fails += gates.g3_noise(groups, [1.0] * 10).status != "pass"
    assert fails <= 20  # two-sided 5% test: 10 expected


def test_g3_not_testable_without_replicates():
    assert gates.g3_noise([np.array([1.0]), np.array([2.0])], [0.1, 0.1]).status == "not testable"
    assert gates.g3_noise([], []).status == "not testable"


# ----------------------------------------------------------------------------------------------
# G4, G5
# ----------------------------------------------------------------------------------------------


def g4_setup(seed, shift=0.0):
    plist, data, z, blocks = g2_setup(seed)
    block = np.concatenate(blocks[:12])  # the "coarsest level": 12 runs, one output each
    z = np.asarray(z).copy()
    z[block] += shift
    return plist, data, z, block


def test_g4_passes_when_the_coarsest_level_follows_the_model_and_fails_when_it_does_not():
    plist, data, z, block = g4_setup(0)
    zz = jnp.stack([jnp.asarray(z)] * 3)
    r = gates.g4_coarsest_level(stack([plist[0]] * 3), data, zz, CFG, block)
    assert r.name == "G4" and r.status == "pass", r.stats
    assert r.stats["p"] == pytest.approx(stats.chi2.sf(r.stats["statistic"], len(block)), rel=1e-12)
    assert r.stats["dof"] == len(block)
    plist, data, z, block = g4_setup(0, shift=1.0)  # a smooth-free offset of the coarsest level only
    r = gates.g4_coarsest_level(stack([plist[0]] * 3), data, jnp.stack([jnp.asarray(z)] * 3), CFG, block)
    assert r.status == "fail" and r.stats["advice"] == "remove the coarsest level"


def test_g4_whitens_with_the_pooled_predictive_covariance_of_the_block():
    plist, data, z, block = g4_setup(1)
    p_a, p_b = plist[0], plist[0]._replace(c0=plist[0].c0 + 0.3)
    zz = jnp.stack([jnp.asarray(z)] * 2)
    r = gates.g4_coarsest_level(stack([p_a, p_b]), data, zz, CFG, block)
    m_a, c_a = gates.block_loo(p_a, data, jnp.asarray(z), CFG, block)
    m_b, c_b = gates.block_loo(p_b, data, jnp.asarray(z), CFG, block)
    d = np.asarray(m_a - m_b)
    c_pool = 0.5 * (np.asarray(c_a) + np.asarray(c_b)) + 0.25 * np.outer(d, d)
    mbar = 0.5 * np.asarray(m_a + m_b)
    e = np.linalg.solve(np.linalg.cholesky(c_pool), np.asarray(z)[block] - mbar)
    assert r.stats["statistic"] == pytest.approx(float(e @ e), rel=1e-8)


def test_g4_not_testable_without_a_block():
    plist, data, z, _ = g4_setup(0)
    r = gates.g4_coarsest_level(stack([plist[0]]), data, jnp.asarray(z)[None], CFG, np.zeros(0, dtype=int))
    assert r.status == "not testable"


def test_g5_monotone():
    inc = np.array([0.0, 0.1, 0.1, 0.5, 2.0])
    assert gates.g5_monotone(inc, "increasing").status == "pass"
    dip = np.array([0.0, 0.1, 0.09, 0.5, 2.0])
    r = gates.g5_monotone(dip, "increasing")
    assert r.status == "fail" and r.stats["n_violations"] == 1
    assert gates.g5_monotone(-inc, "decreasing").status == "pass"
    assert gates.g5_monotone(inc, "decreasing").status == "fail"
    assert gates.g5_monotone(inc, None).status == "not testable"


# ----------------------------------------------------------------------------------------------
# G6
# ----------------------------------------------------------------------------------------------


def test_g6_gaussian_draws_pass():
    rng = np.random.default_rng(30)
    draws = 3.0 + 2.0 * rng.standard_normal((60000, 3))
    r = gates.g6_shape(draws, np.full(60000, 1 / 60000))
    assert r.name == "G6" and r.status == "pass", r.stats
    # the 16/84 half-width of a Gaussian is 0.9945 sd, so the tails differ from m +- 1.96 sigma by 0.011 sigma
    assert r.stats["median_rel_diff"] < 0.05


def test_g6_skewed_lognormal_fails():
    rng = np.random.default_rng(31)
    draws = np.exp(1.0 * rng.standard_normal((60000, 2)))
    r = gates.g6_shape(draws, np.full(60000, 1 / 60000))
    assert r.status == "fail"
    assert r.stats["median_rel_diff"] > 0.2


def test_g6_judges_the_median_over_points_not_the_worst_point():
    rng = np.random.default_rng(33)
    good = rng.standard_normal((60000, 9))
    wild = np.exp(rng.standard_normal((60000, 1)))  # one skewed point of ten
    r = gates.g6_shape(np.hstack([good, wild]), np.full(60000, 1 / 60000))
    assert r.status == "pass" and np.max(r.stats["rel_diff"]) > 0.2


def test_g6_respects_the_weights():
    rng = np.random.default_rng(32)
    gauss = rng.standard_normal((40000, 1))
    wild = 50.0 + 10.0 * rng.standard_normal((40000, 1))
    draws = np.vstack([gauss, wild])
    w = np.concatenate([np.full(40000, 1 / 40000), np.zeros(40000)])  # all weight on the Gaussian half
    assert gates.g6_shape(draws, w).status == "pass"


def test_g6_not_testable_for_zero_spread():
    assert gates.g6_shape(np.ones((100, 1)), np.full(100, 0.01)).status == "not testable"


# ----------------------------------------------------------------------------------------------
# G7: conjugate Gaussian toy, prior N(0, tau^2), n observations of unit noise
# ----------------------------------------------------------------------------------------------


def toy_posterior(ybar, n, tau, S, seed):
    """Exact posterior draws of theta ~ N(0, tau^2) given the mean ybar of n unit-noise observations."""
    prec = n + 1.0 / tau**2
    rng = np.random.default_rng(seed)
    theta = ybar * n / prec + rng.standard_normal(S) / np.sqrt(prec)
    return theta


def toy_ratio_fn(tau):
    def fn(scenario, theta):
        t_new = tau * scenario  # scenario = scale factor on the prior sd
        return stats.norm.logpdf(theta, 0, t_new) - stats.norm.logpdf(theta, 0, tau)

    return fn


def posterior_moments(ybar, n, tau):
    prec = n + 1.0 / tau**2
    return ybar * n / prec, 1.0 / np.sqrt(prec)


def test_g7_a_prior_change_that_does_not_matter_passes():
    S, n, tau = 20000, 400, 1.0
    theta = toy_posterior(0.7, n, tau, S, 40)
    r = gates.g7_prior(
        toy_ratio_fn(tau), theta[:, None], np.full(S, 1 / S), transforms.get("identity"), [0.5, 2.0], theta
    )
    assert r.name == "G7" and r.status == "pass", r.stats
    # analytic reference: halving the prior sd moves the posterior mean by < 0.5 sigma_epi
    m0, s0 = posterior_moments(0.7, n, tau)
    m1, _ = posterior_moments(0.7, n, 0.5 * tau)
    assert abs(m1 - m0) < 0.5 * s0
    assert r.stats["ess"][0.5] > 400 and r.stats["ess"][2.0] > 400


def test_g7_prior_dominated_toy_fails_and_matches_the_analytic_shift():
    S, n, tau, ybar = 40000, 1, 1.0, 3.0
    theta = toy_posterior(ybar, n, tau, S, 41)
    r = gates.g7_prior(
        toy_ratio_fn(tau), theta[:, None], np.full(S, 1 / S), transforms.get("identity"), [0.5], theta
    )
    assert r.status == "fail" and r.stats["verdict"] == "prior-dominated"
    m0, s0 = posterior_moments(ybar, n, tau)
    m1, s1 = posterior_moments(ybar, n, 0.5 * tau)
    assert r.stats["m_shift"][0.5][0] == pytest.approx((m1 - m0) / s0, abs=0.05)
    assert r.stats["sigma_change"][0.5][0] == pytest.approx(s1 / s0 - 1.0, abs=0.05)


def test_g7_low_ess_asks_for_a_refit():
    S, n, tau = 300, 400, 1.0  # 300 draws can never reach ESS 400
    theta = toy_posterior(0.7, n, tau, S, 42)
    r = gates.g7_prior(
        toy_ratio_fn(tau), theta[:, None], np.full(S, 1 / S), transforms.get("identity"), [0.5, 2.0], theta
    )
    assert r.status == "not testable" and r.stats["verdict"] == "refit needed"


def test_g7_extreme_prior_change_has_low_ess_even_with_many_draws():
    S, n, tau = 20000, 400, 1.0
    theta = toy_posterior(0.7, n, tau, S, 43)
    r = gates.g7_prior(
        toy_ratio_fn(tau), theta[:, None], np.full(S, 1 / S), transforms.get("identity"), [0.001], theta
    )
    assert r.status == "not testable" and r.stats["ess"][0.001] < 400


def test_g7_a_valid_scenario_that_moves_the_answer_fails_even_if_another_needs_a_refit():
    S, n, tau, ybar = 40000, 1, 1.0, 3.0
    theta = toy_posterior(ybar, n, tau, S, 44)
    r = gates.g7_prior(
        toy_ratio_fn(tau), theta[:, None], np.full(S, 1 / S), transforms.get("identity"), [0.5, 0.001], theta
    )
    assert r.stats["ess"][0.001] < 400
    assert r.status == "fail"
