"""Stacking weights of candidate structures (spec 2.7): hold-out, scores, Dirichlet-penalised stacking.

Posterior draws are built directly from known ModelParams (no MCMC). References are NumPy computations.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp
from test_acquisition import CFG, make_data, make_params, simulate_z, stack

import gcbml  # noqa: F401
from gcbml import model, transforms
from gcbml import stacking as stk
from gcbml.acquisition import StructurePosterior

# ----------------------------------------------------------------------------------------------
# stack_weights
# ----------------------------------------------------------------------------------------------


def objective(w, scores, alpha):
    w = np.asarray(w)
    return float(np.sum(logsumexp(np.log(w)[:, None] + scores, axis=0)) + (alpha - 1.0) * np.sum(np.log(w)))


def gradient(w, scores, alpha):
    mix = np.exp(logsumexp(np.log(w)[:, None] + scores, axis=0))
    return np.exp(scores) @ (1.0 / mix) + (alpha - 1.0) / w


def test_dominating_structure_gets_the_dirichlet_limit():
    # structure 0 beats the others by 60 nats on each of n rows: the likelihood weight of the others is
    # ~ e^-60, so w_0 = (n + alpha - 1) / (n + M (alpha - 1)), the posterior mode of the Dirichlet(alpha)
    # prior after n observations that all belong to structure 0.
    n, M, alpha = 40, 3, 2.0
    scores = np.zeros((M, n))
    scores[0] += 60.0
    w = stk.stack_weights(scores, alpha)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert w[0] == pytest.approx((n + alpha - 1) / (n + M * (alpha - 1)), abs=1e-9)
    assert w[1] == pytest.approx((alpha - 1) / (n + M * (alpha - 1)), abs=1e-9)
    assert w[0] > 0.95


def test_identical_scores_give_equal_weights():
    scores = np.tile(np.random.default_rng(0).normal(size=(1, 25)), (4, 1))
    assert np.allclose(stk.stack_weights(scores), 0.25, atol=1e-9)


def test_single_structure_has_weight_one():
    assert np.allclose(stk.stack_weights(np.random.default_rng(1).normal(size=(1, 7))), [1.0])


@pytest.mark.parametrize("alpha", [2.0, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stack_weights_satisfy_the_kkt_conditions(seed, alpha):
    # concave problem: at the optimum the gradient components are equal on the support and no larger off it
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(4, 30)) * np.array([[1.0], [1.5], [2.0], [0.5]]) - np.array(
        [[0], [1], [3], [20]]
    )
    w = stk.stack_weights(scores, alpha)
    assert w.sum() == pytest.approx(1.0, abs=1e-12) and np.all(w >= 0)
    supp = w > 1e-8
    assert supp.sum() >= 1
    g = gradient(np.where(supp, w, 1.0), scores, alpha)
    lam = g[supp]
    # alpha = 1 puts the optimum on the boundary, where the EM fixed point converges only sublinearly
    tol = 1e-6 if alpha > 1 else 1e-1
    assert np.ptp(lam) < tol * np.abs(lam).mean()
    # off the support, the marginal value of adding weight must not exceed lambda
    if (~supp).any():
        mix = np.exp(logsumexp(np.log(np.maximum(w, 1e-300))[:, None] + scores, axis=0))
        g_off = (np.exp(scores) @ (1.0 / mix))[~supp]
        assert np.all(g_off <= lam.mean() * (1 + 1e-6))


def test_stack_weights_match_grid_search_for_two_structures():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(2, 20)) + np.array([[0.0], [-0.3]])
    alpha = 2.0
    grid = np.linspace(1e-6, 1 - 1e-6, 200001)
    vals = [objective([1 - a, a], scores, alpha) for a in grid[::50]]
    a_best = grid[::50][int(np.argmax(vals))]
    w = stk.stack_weights(scores, alpha)
    assert w[1] == pytest.approx(a_best, abs=2e-4)
    assert objective(w, scores, alpha) >= max(vals) - 1e-9


def test_stack_weights_match_grid_search_for_three_structures():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(3, 25)) + np.array([[0.0], [-0.2], [-0.5]])
    alpha = 2.0
    step = 0.005
    best, arg = -np.inf, None
    for a in np.arange(step, 1.0, step):
        for b in np.arange(step, 1.0 - a, step):
            c = 1.0 - a - b
            if c <= 0:
                continue
            v = objective([a, b, c], scores, alpha)
            if v > best:
                best, arg = v, (a, b, c)
    w = stk.stack_weights(scores, alpha)
    assert np.allclose(w, arg, atol=1.5 * step)
    assert objective(w, scores, alpha) >= best - 1e-9


# ----------------------------------------------------------------------------------------------
# a small world: three levels, hold out the finest
# ----------------------------------------------------------------------------------------------

NX, N_PAD = 8, 48
HBARS = (1.0, 0.5, 0.25)


def world(seed=3, nx=NX, truth="identity", sigma_mu=1.0):
    """Data at three levels on nx sites; z simulated from the model with known params (exp if lognormal)."""
    xs = np.linspace(0.05, 0.95, nx)
    X = np.concatenate([xs[:, None]] * 3)
    H = np.concatenate([np.full((nx, 1), h) for h in HBARS])
    data = make_data(X, H, N_PAD)
    p_true = make_params(N_PAD, sigma_mu=sigma_mu)
    z = np.asarray(simulate_z(seed, p_true, data, CFG))
    y = np.exp(z) if truth == "log" else z
    n = 3 * nx
    levels = np.concatenate([np.repeat([0, 1, 2], nx), np.full(N_PAD - n, -1)])
    return dataclasses.replace(data, y=y), levels, z


def plist_around(base=1.4, k=3):
    return [make_params(N_PAD, p=base + 0.1 * s, c0=0.5 - 0.05 * s, ell=0.3 + 0.02 * s) for s in range(k)]


def structure(plist, y, tf_name):
    tf = transforms.get(tf_name)
    z = tf.forward(jnp.asarray(y))
    return StructurePosterior(
        params=stack(plist), z=jnp.stack([z] * len(plist)), cfg=CFG, transform=tf, weight=1.0
    )


# ----------------------------------------------------------------------------------------------
# holdout
# ----------------------------------------------------------------------------------------------


def test_holdout_removes_the_finest_level_from_the_training_rows():
    data, levels, _ = world()
    train, rows = stk.holdout(data, levels, finest=2)
    assert np.array_equal(rows, np.arange(2 * NX, 3 * NX))
    assert train.mask.sum() == 2 * NX and not train.mask[rows].any()
    # untouched arrays: the held-out rows keep their coordinates, so they can be scored
    assert (
        np.array_equal(train.X, data.X)
        and np.array_equal(train.H, data.H)
        and np.array_equal(train.y, data.y)
    )
    assert train.mask[: 2 * NX].all()


def test_holdout_drops_censored_rows_from_both_sets():
    data, levels, _ = world()
    cens = np.asarray(data.censored).copy()
    cens[2 * NX + 1] = True
    data = dataclasses.replace(data, censored=cens)
    train, rows = stk.holdout(data, levels, finest=2)
    assert (2 * NX + 1) not in rows and len(rows) == NX - 1
    assert not train.mask[2 * NX + 1]


# ----------------------------------------------------------------------------------------------
# log_scores
# ----------------------------------------------------------------------------------------------


def reference_scores(plist, data, z, log_jac, train_rows, held_rows):
    """Pooled-mixture log density of the held-out outputs, from dense NumPy algebra on the joint Gaussian."""
    b0, bsd = (np.asarray(a) for a in CFG.beta_prior)
    per_draw = []
    for p in plist:
        S = model._setup(p, data, CFG)  # data: every row real, so K holds train and held-out rows
        Kt = np.asarray(S.K + (S.A * bsd**2) @ S.A.T)
        mu = np.asarray(S.rho0 + S.A @ b0)
        Ktt = Kt[np.ix_(train_rows, train_rows)]
        Kht = Kt[np.ix_(held_rows, train_rows)]
        m = mu[held_rows] + Kht @ np.linalg.solve(Ktt, z[train_rows] - mu[train_rows])
        V = Kt[np.ix_(held_rows, held_rows)] - Kht @ np.linalg.solve(Ktt, Kht.T)
        per_draw.append(stats.norm.logpdf(z[held_rows], m, np.sqrt(np.diag(V))))
    return logsumexp(np.array(per_draw), axis=0) - np.log(len(plist)) + log_jac


@pytest.mark.parametrize("tf_name", ["identity", "log"])
def test_log_scores_equal_the_pooled_mixture_density_with_the_jacobian(tf_name):
    data, levels, z0 = world(truth="log")
    n = 3 * NX
    data_full = dataclasses.replace(data, mask=np.arange(N_PAD) < n)
    plist = plist_around()
    train, rows = stk.holdout(data, levels, finest=2)
    sp = structure(plist, data.y, tf_name)
    got = np.asarray(stk.log_scores([sp], train, rows))
    assert got.shape == (1, NX)
    tf = transforms.get(tf_name)
    zz = np.asarray(tf.forward(jnp.asarray(data.y)))
    log_jac = np.asarray(tf.log_abs_jac(jnp.asarray(data.y)))[rows]
    train_rows = np.arange(2 * NX)
    ref = reference_scores(plist, data_full, zz, log_jac, train_rows, rows)
    assert np.allclose(got[0], ref, atol=1e-8, rtol=0)
    if tf_name == "log":
        assert np.allclose(log_jac, -np.log(data.y[rows]))


def test_log_scores_stacks_structures_in_order():
    data, levels, _ = world()
    train, rows = stk.holdout(data, levels, finest=2)
    a = structure(plist_around(1.4), data.y, "identity")
    b = structure(plist_around(2.0), data.y, "identity")
    both = np.asarray(stk.log_scores([a, b], train, rows))
    assert both.shape == (2, NX)
    assert np.allclose(both[0], np.asarray(stk.log_scores([a], train, rows))[0], atol=1e-10)
    assert np.allclose(both[1], np.asarray(stk.log_scores([b], train, rows))[0], atol=1e-10)


def test_jacobian_makes_a_lognormal_truth_prefer_the_log_structure():
    # y = exp(z), z from the model. Identity structure: a Gaussian on y; log structure: the true model.
    # Scores are densities of y, so they are comparable; the log structure must win on average, and it
    # must win by more than the Jacobian term alone (which is not what separates them: it is the density).
    wins, diffs = 0, []
    for seed in range(6):
        data, levels, _ = world(seed=seed, nx=12, truth="log", sigma_mu=0.8)
        train, rows = stk.holdout(data, levels, finest=2)
        sc = np.asarray(
            stk.log_scores(
                [
                    structure([make_params(N_PAD, sigma_mu=0.8)], data.y, "log"),
                    structure([make_params(N_PAD, sigma_mu=0.8)], data.y, "identity"),
                ],
                train,
                rows,
            )
        )
        diffs.append(sc[0].sum() - sc[1].sum())
        wins += sc[0].sum() > sc[1].sum()
    assert wins == 6, diffs
    # and the stacking weight follows
    assert stk.stack_weights(sc)[0] > 0.8


# ----------------------------------------------------------------------------------------------
# split_sites, cross_fitted_weights
# ----------------------------------------------------------------------------------------------


def sites(n_sites, reps=3):
    u = np.linspace(0.0, 1.0, n_sites)[:, None]
    return np.repeat(u, reps, axis=0)


def test_split_sites_splits_distinct_u_and_keeps_replicates_together():
    import jax

    X = sites(14)
    out = stk.split_sites(jax.random.key(0), X)
    assert out is not None
    A, B = out
    assert len(set(A) & set(B)) == 0 and sorted(np.concatenate([A, B])) == list(range(len(X)))
    ua, ub = {tuple(r) for r in X[A]}, {tuple(r) for r in X[B]}
    assert len(ua) == 7 and len(ub) == 7 and not (ua & ub)


def test_split_sites_is_deterministic_given_the_key_and_varies_with_it():
    import jax

    X = sites(14)
    a1, b1 = stk.split_sites(jax.random.key(5), X)
    a2, b2 = stk.split_sites(jax.random.key(5), X)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    a3, _ = stk.split_sites(jax.random.key(6), X)
    assert not np.array_equal(a1, a3)


def test_split_sites_needs_six_sites_per_half():
    import jax

    assert stk.split_sites(jax.random.key(0), sites(11)) is None  # halves of 5 and 6
    assert stk.split_sites(jax.random.key(0), sites(12)) is not None  # 6 and 6
    assert stk.split_sites(jax.random.key(0), sites(12), min_sites=7) is None


def test_cross_fitted_weights_use_each_half_for_the_other_calibration():
    import jax

    rng = np.random.default_rng(7)
    n_sites = 14
    site_of_row = np.repeat(np.arange(n_sites), 3)
    scores = rng.normal(size=(3, site_of_row.size)) + np.array([[0.0], [-0.3], [-1.0]])
    cf = stk.cross_fitted_weights(jax.random.key(1), scores, site_of_row)
    assert cf.testable
    assert not set(site_of_row[cf.idx_A]) & set(site_of_row[cf.idx_B])
    assert np.allclose(cf.w_A, stk.stack_weights(scores[:, cf.idx_A]), atol=1e-12)
    assert np.allclose(cf.w_B, stk.stack_weights(scores[:, cf.idx_B]), atol=1e-12)
    cf2 = stk.cross_fitted_weights(jax.random.key(1), scores, site_of_row)
    assert np.array_equal(cf.idx_A, cf2.idx_A) and np.allclose(cf.w_A, cf2.w_A, atol=0)


def test_cross_fitted_weights_with_too_few_sites_are_equal_and_not_testable():
    import jax

    rng = np.random.default_rng(8)
    site_of_row = np.repeat(np.arange(11), 3)
    scores = rng.normal(size=(3, site_of_row.size)) + np.array([[5.0], [0.0], [-5.0]])
    cf = stk.cross_fitted_weights(jax.random.key(1), scores, site_of_row)
    assert not cf.testable
    assert np.allclose(cf.w_A, 1 / 3) and np.allclose(cf.w_B, 1 / 3)
