"""Step 5: sigma_epi on the physical scale, hinge/softmax H, fantasies, reweighting, budget, MR-SUR, batches.

The posterior draws are built directly from known ModelParams (no MCMC): the world is one input x, one
resolution component, hbar in (0, 1], and a Yi-h model with known parameters.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize, stats
from test_model_joint_new import extend

import gcbml  # noqa: F401
from gcbml import acquisition as acq
from gcbml import model, transforms
from gcbml.acquisition import Candidate, StructurePosterior
from gcbml.data import PaddedData
from gcbml.kernels import DeltaParams
from gcbml.model import ModelConfig, ModelParams

Z84 = float(stats.norm.ppf(0.84))  # the 16/84 quantile half-width of a Gaussian is Z84 * sd, not sd

# ----------------------------------------------------------------------------------------------
# A small world
# ----------------------------------------------------------------------------------------------


def make_params(n_pad, p=1.4, c0=0.5, c1=0.2, sigma_mu=1.0, ell=0.3, dsig=0.15, nv=1e-3):
    return ModelParams(
        sigma_mu=jnp.asarray(sigma_mu),
        ell_mu=jnp.asarray([ell]),
        c0=jnp.asarray([c0]),
        c1=jnp.asarray([c1]),
        P=jnp.full((n_pad, 1), p),
        delta=DeltaParams(
            sigma=jnp.asarray([dsig]),
            ell_x=jnp.asarray([[0.4]]),
            ell_h=jnp.asarray([0.8]),
            gamma=jnp.asarray([0.5]),
        ),
        noise_var=jnp.full((n_pad,), nv),
        ell_v=jnp.asarray(0.3),
    )


def stack(plist):
    return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *plist)


def make_data(X, H, n_pad):
    n = len(X)
    pad = n_pad - n
    return PaddedData(
        X=np.vstack([X, np.repeat(X[:1], pad, 0)]),
        H=np.vstack([H, np.repeat(H[:1], pad, 0)]),
        y=np.zeros(n_pad),
        censored=np.zeros(n_pad, bool),
        run=np.concatenate([np.arange(n), np.full(pad, -1)]),
        mask=np.concatenate([np.ones(n, bool), np.zeros(pad, bool)]),
    )


def simulate_z(seed, params, data, cfg):
    """One draw of the outputs z from the model (beta drawn from its Gaussian prior)."""
    S = model._setup(params, data, cfg)
    b0, bsd = model._beta_prior(cfg)
    Kt = np.asarray(S.K + (S.A * bsd**2) @ S.A.T)
    mean = np.asarray(S.rho0 + S.A @ b0)
    real = np.asarray(data.mask)
    z = np.zeros(len(real))
    z[real] = np.random.default_rng(seed).multivariate_normal(mean[real], Kt[np.ix_(real, real)])
    return jnp.asarray(z)


CFG = ModelConfig(h_kernel="twy2", beta_prior=((0.0,), (1.0,)))
CFG_FLAT = ModelConfig(h_kernel="twy2", beta_prior=None)


def world(plist=None, weights=None, cfg=CFG, transform="identity", n_levels=(1.0, 0.5), nx=8, seed=1):
    """Structures with one draw per entry of ``plist``; data at levels n_levels on a grid of nx sites."""
    xs = np.linspace(0.05, 0.95, nx)
    X = np.concatenate([xs[:, None]] * len(n_levels))
    H = np.concatenate([np.full((nx, 1), h) for h in n_levels])
    n_pad = 24
    data = make_data(X, H, n_pad)
    plist = plist or [make_params(n_pad)]
    z = simulate_z(seed, plist[0], data, cfg if cfg.beta_prior is not None else CFG)
    tf = transforms.get(transform)
    st = StructurePosterior(
        params=stack(plist),
        z=jnp.stack([z] * len(plist)),
        cfg=cfg,
        transform=tf,
        weight=1.0,
    )
    return (st,), data


def cand(x, h, cost=1.0, cap=None, nv=1e-3, level=1):
    return Candidate(
        Xa=np.array([[x]]),
        hbar=np.array([h]),
        levels=np.array([level]),
        cost_mean=cost,
        cost_cap=cost * 1.5 if cap is None else cap,
        noise_var=np.array([nv]),
    )


XS = np.linspace(0.0, 1.0, 9)[:, None]


# ----------------------------------------------------------------------------------------------
# sigma_epi on the physical scale
# ----------------------------------------------------------------------------------------------


def per_draw_moments(structs, data, Xs):
    st = structs[0]
    f = jax.vmap(lambda p, z: model.predict_mu(p, data, z, st.cfg, jnp.asarray(Xs)))
    m, v = f(st.params, st.z)
    return np.asarray(m), np.asarray(v)


def test_sigma_epi_single_draw_identity_is_gaussian_quantile_halfwidth():
    structs, data = world()
    m, v = per_draw_moments(structs, data, XS)
    got = np.asarray(acq.sigma_epi_physical(structs, data, XS))
    np.testing.assert_allclose(got, Z84 * np.sqrt(v[0]), rtol=1e-9)


def mixture_halfwidth_brentq(m, sd, w, inv=lambda z: z):
    """Half-width of the physical mixture by root-finding on its exact CDF (independent of the code)."""
    out = []
    for j in range(m.shape[1]):
        mj, sj = m[:, j], sd[:, j]

        def cdf(q, mj=mj, sj=sj):
            return np.sum(w * stats.norm.cdf((q - mj) / sj))

        qs = [optimize.brentq(lambda q, p=p, cdf=cdf: cdf(q) - p, -50, 50, xtol=1e-13) for p in (0.16, 0.84)]
        out.append(0.5 * abs(inv(qs[1]) - inv(qs[0])))
    return np.array(out)


def two_draw_world(transform="identity", cfg=CFG, c0s=(0.0, 1.5)):
    return world([make_params(24, c0=c) for c in c0s], cfg=cfg, transform=transform)


def test_sigma_epi_two_component_mixture_matches_exact_cdf_and_sampling():
    structs, data = two_draw_world()
    m, v = per_draw_moments(structs, data, XS)
    assert np.abs(m[0] - m[1]).max() > 0.1  # the draws really disagree
    w = np.array([0.5, 0.5])
    got = np.asarray(acq.sigma_epi_physical(structs, data, XS))
    exact = mixture_halfwidth_brentq(m, np.sqrt(v), w)
    np.testing.assert_allclose(got, exact, rtol=1e-8)
    rng = np.random.default_rng(0)
    N = 2_000_000
    pick = rng.integers(0, 2, size=N)
    samp = m[pick] + np.sqrt(v[pick]) * rng.standard_normal((N, m.shape[1]))
    mc = 0.5 * (np.quantile(samp, 0.84, axis=0) - np.quantile(samp, 0.16, axis=0))
    np.testing.assert_allclose(got, mc, rtol=3e-3)


@pytest.mark.parametrize("name", ["log", "reciprocal"])
def test_sigma_epi_log_and_reciprocal_match_inverse_transform_sampling(name):
    cfg = ModelConfig(h_kernel="twy2", beta_prior=((4.0,), (0.5,)), increasing=(name == "log"))
    structs, data = two_draw_world(transform=name, cfg=cfg, c0s=(0.0, 0.4))
    m, v = per_draw_moments(structs, data, XS)
    assert m.min() - 8 * np.sqrt(v.max()) > 0.5  # Lambda units well inside the domain
    got = np.asarray(acq.sigma_epi_physical(structs, data, XS))
    inv = np.exp if name == "log" else (lambda z: 1.0 / z)
    exact = mixture_halfwidth_brentq(m, np.sqrt(v), np.array([0.5, 0.5]), inv)
    np.testing.assert_allclose(got, exact, rtol=1e-7)
    rng = np.random.default_rng(1)
    N = 2_000_000
    pick = rng.integers(0, 2, size=N)
    y = inv(m[pick] + np.sqrt(v[pick]) * rng.standard_normal((N, m.shape[1])))
    mc = 0.5 * (np.quantile(y, 0.84, axis=0) - np.quantile(y, 0.16, axis=0))
    np.testing.assert_allclose(got, mc, rtol=5e-3)


def test_sigma_epi_pools_structures_with_different_transforms_on_the_physical_scale():
    (s_id,), data = world([make_params(24)])
    cfg_log = ModelConfig(h_kernel="twy2", beta_prior=((1.0,), (0.5,)))
    (s_log,), data_l = world([make_params(24, c0=0.2)], cfg=cfg_log, transform="log")
    assert np.array_equal(data.X, data_l.X)
    s_log = s_log._replace(weight=0.3)
    s_id = s_id._replace(weight=0.7, z=s_id.z)
    got = np.asarray(acq.sigma_epi_physical((s_id, s_log), data, XS))
    m0, v0 = per_draw_moments((s_id,), data, XS)
    m1, v1 = per_draw_moments((s_log,), data, XS)
    for j in range(XS.shape[0]):

        def cdf(y, j=j):
            lg = stats.norm.cdf((np.log(y) - m1[0, j]) / np.sqrt(v1[0, j])) if y > 0 else 0.0
            return 0.7 * stats.norm.cdf((y - m0[0, j]) / np.sqrt(v0[0, j])) + 0.3 * lg

        qs = [optimize.brentq(lambda y, p=p: cdf(y) - p, -50.0, 1e3, xtol=1e-13) for p in (0.16, 0.84)]
        np.testing.assert_allclose(got[j], 0.5 * (qs[1] - qs[0]), rtol=1e-6)


# ----------------------------------------------------------------------------------------------
# H
# ----------------------------------------------------------------------------------------------


def test_H_hinge_zero_when_all_below_eps_positive_otherwise_and_hand_value():
    eps = 0.5
    assert float(acq.H_value(jnp.array([0.1, 0.5, 0.3]), eps)) == 0.0
    sig = jnp.array([0.1, 1.0, 0.75])  # (1/0.25 - 1)... sigma^2/eps^2 = 0.04, 4, 2.25
    assert float(acq.H_value(sig, eps)) > 0.0
    np.testing.assert_allclose(float(acq.H_value(sig, eps)), (4.0 - 1.0) + (2.25 - 1.0), rtol=1e-12)
    # eps per site
    np.testing.assert_allclose(
        float(acq.H_value(sig, jnp.array([1.0, 1.0, 0.5]))), (0.0) + 0.0 + (2.25 - 1.0), rtol=1e-12
    )


def test_H_softmax_is_at_least_max_and_tends_to_max():
    sig, eps = jnp.array([0.3, 0.9, 0.6]), 0.5
    r = np.asarray(sig**2 / eps**2)
    vals = [float(acq.H_value(sig, eps, mode="softmax", beta=b)) for b in (1.0, 5.0, 20.0, 100.0, 1000.0)]
    assert all(v >= r.max() - 1e-12 for v in vals)
    assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))
    assert abs(vals[-1] - r.max()) < 1e-2
    np.testing.assert_allclose(vals[2], np.log(np.exp(20.0 * r).sum()) / 20.0, rtol=1e-12)
    with pytest.raises(ValueError):
        acq.H_value(sig, eps, mode="nope")


# ----------------------------------------------------------------------------------------------
# Fantasies
# ----------------------------------------------------------------------------------------------


def deterministic_gain(structs, data, c, Xs, eps, mode="hinge"):
    """H before - H after appending the candidate's rows WITHOUT a value (one draw): predict_mu on the
    enlarged data, in which z of the new rows is irrelevant for the variance."""
    st = structs[0]
    p = jax.tree_util.tree_map(lambda a: a[0], st.params)
    z = st.z[0]
    m = len(c.Xa)
    Pn = np.tile(np.asarray(p.P[0]), (m, 1))
    Hn = np.tile(np.asarray(c.hbar)[None, :], (m, 1))
    p2, d2, z2 = extend(p, data, z, c.Xa, Hn, Pn, np.asarray(c.noise_var), np.zeros(m), run_id=999)
    _, v1 = model.predict_mu(p, data, z, st.cfg, jnp.asarray(Xs))
    _, v2 = model.predict_mu(p2, d2, z2, st.cfg, jnp.asarray(Xs))
    H1 = acq.H_value(Z84 * jnp.sqrt(v1), eps, mode)
    H2 = acq.H_value(Z84 * jnp.sqrt(v2), eps, mode)
    return float(H1 - H2)


@pytest.mark.parametrize("cfg", [CFG, CFG_FLAT])
def test_variance_only_one_draw_identity_gain_equals_deterministic_gain(cfg):
    structs, data = world(cfg=cfg)
    sig = np.asarray(acq.sigma_epi_physical(structs, data, XS))
    eps = 0.5 * float(sig.max())
    for c in (cand(0.5, 1.0), cand(0.2, 0.5), cand(0.7, 0.25)):
        gain, se = acq.expected_gain(jax.random.key(0), structs, data, c, XS, eps, 16)
        det = deterministic_gain(structs, data, c, XS, eps)
        assert det > 0
        np.testing.assert_allclose(gain, det, rtol=1e-7)
        assert se < 1e-8 * det


def test_variance_only_posterior_variance_does_not_depend_on_the_value_many_draws():
    """The conditioned variance of every draw equals predict_mu on the enlarged data, for any new value."""
    structs, data = world([make_params(24, p=pp, c0=c) for pp, c in ((0.8, 0.3), (2.0, 0.9), (1.4, 0.0))])
    c = cand(0.6, 0.25)
    pool = acq._build_pool(structs, data, XS, [c])
    for ya in (-3.0, 0.4, 7.0):
        mean_c, var_c = acq._condition_values(pool, 0, np.array([ya]))
        st = structs[0]
        for s in range(3):
            p = jax.tree_util.tree_map(lambda a, s=s: a[s], st.params)
            Pn = np.asarray(p.P[:1])
            p2, d2, z2 = extend(
                p,
                data,
                st.z[s],
                c.Xa,
                np.asarray(c.hbar)[None, :],
                Pn,
                np.asarray(c.noise_var),
                np.array([ya]),
                5,
            )
            m2, v2 = model.predict_mu(p2, d2, z2, st.cfg, jnp.asarray(XS))
            np.testing.assert_allclose(var_c[s], v2, rtol=1e-6, atol=1e-10)
            np.testing.assert_allclose(mean_c[s], m2, rtol=1e-6, atol=1e-8)


def reweight_world():
    """Two draws that differ only in the order p (0.8 and 2.0), data at hbar = 1 and 0.5, equal weights.

    Equal weights are imposed (stacking weights are held fixed); the data would favour p = 0.8 by a factor of
    about 20 in marginal likelihood, and the two draws disagree about mu(x) by about 0.4, because rho0 and
    rho1 at hbar = 0.5 depend on p through hbar^p.
    """
    plist = [make_params(24, p=pp, c0=1.0, c1=0.0, dsig=0.3, nv=1e-3) for pp in (0.8, 2.0)]
    return world(plist, n_levels=(1.0, 0.5), nx=8)


def test_data_at_one_level_cannot_tell_orders_apart():
    """At hbar = 1 every order gives hbar^p = 1: the likelihood is identical, whatever p."""
    plist = [make_params(24, p=pp, c0=1.0, c1=0.0, dsig=0.3, nv=1e-3) for pp in (0.8, 2.0)]
    (st,), data = world(plist, n_levels=(1.0,), nx=8)
    lm = [
        float(
            model.log_marginal(jax.tree_util.tree_map(lambda a, s=s: a[s], st.params), data, st.z[s], st.cfg)
        )
        for s in (0, 1)
    ]
    assert abs(lm[0] - lm[1]) < 1e-9


def test_reweighting_by_the_fantasy_likelihood_raises_the_value_of_a_finer_unprobed_level():
    structs, data = reweight_world()
    m, _ = per_draw_moments(structs, data, XS)
    assert np.abs(m[0] - m[1]).max() > 0.3
    sig0 = np.asarray(acq.sigma_epi_physical(structs, data, XS))
    eps = 0.5 * float(sig0.max())
    key = jax.random.key(3)
    ratio = {}
    for h in (0.25, 0.5):  # unprobed finer level, and a probed level
        c = cand(0.5, h, nv=1e-3)
        rw = acq.expected_gain_detail(key, structs, data, c, XS, eps, 128, ess_min=1.0)
        no = acq.expected_gain_detail(key, structs, data, c, XS, eps, 128, ess_min=np.inf)
        assert rw.fallback_frac == 0.0 and no.fallback_frac == 1.0
        ratio[h] = rw.gain / no.gain
        if h == 0.25:
            assert rw.gain > no.gain + 3.0 * (rw.mc_se + no.mc_se)
    # the probed level cannot resolve p: reweighting adds little there
    assert ratio[0.25] > 1.3 and ratio[0.5] < 1.1


def test_ess_fallback_is_triggered_and_reported_with_the_default_threshold():
    structs, data = reweight_world()
    eps = 0.1
    c_fine = cand(0.5, 0.25, nv=1e-4)
    key = jax.random.key(4)
    default = acq.expected_gain_detail(key, structs, data, c_fine, XS, eps, 32)  # ess_min = 50 > 2 draws
    assert default.fallback_frac == 1.0
    keep = acq.expected_gain_detail(key, structs, data, c_fine, XS, eps, 32, ess_min=np.inf)
    np.testing.assert_allclose(default.gain, keep.gain, rtol=1e-12)  # the fallback IS 'keep the weights'
    some = acq.expected_gain_detail(key, structs, data, c_fine, XS, eps, 32, ess_min=1.5)
    assert 0.0 < some.fallback_frac < 1.0  # a constructed case where some fantasies collapse the weights


# ----------------------------------------------------------------------------------------------
# Budget, MR-SUR, batches
# ----------------------------------------------------------------------------------------------


def one_draw_world(**kw):
    structs, data = world(**kw)
    sig = np.asarray(acq.sigma_epi_physical(structs, data, XS))
    return structs, data, 0.5 * float(sig.max())


def test_candidates_with_cap_above_the_remaining_budget_are_never_chosen():
    structs, data, eps = one_draw_world()
    best = cand(0.5, 0.5, cost=1.0, cap=9.0)  # the most valuable one, but its cap is too high
    cands = [best, cand(0.2, 1.0, cost=1.0, cap=1.5), cand(0.8, 1.0, cost=1.0, cap=1.5)]
    chosen, table = acq.select_batch(
        jax.random.key(0), structs, data, cands, XS, eps, 3, budget_remaining=5.0
    )
    assert 0 not in chosen and len(chosen) == 2
    assert not table.admissible[0] and table.admissible[1] and table.admissible[2]


def test_pending_caps_are_subtracted_from_the_budget():
    structs, data, eps = one_draw_world()
    cands = [cand(0.2, 1.0, cap=3.0), cand(0.8, 1.0, cap=3.0)]
    pend = (cand(0.5, 1.0, cap=4.0),)
    key = jax.random.key(0)

    def run(budget):
        return acq.select_batch(key, structs, data, cands, XS, eps, 2, budget, pending=pend)

    chosen, table = run(6.9)  # 6.9 - 4 = 2.9 < 3: nothing admissible
    assert list(table.admissible) == [False, False] and chosen == []
    chosen, table = run(8.0)  # 4 left: one run fits, and its own cap is reserved before the next pick
    assert list(table.admissible) == [True, True] and len(chosen) == 1
    chosen, _ = run(11.0)  # 7 left: both fit
    assert len(chosen) == 2


def test_mr_sur_ratio_twice_the_cost_same_gain_is_never_preferred_and_matches_eq_11():
    structs, data, eps = one_draw_world()
    a = cand(0.5, 0.5, cost=1.0)
    b = cand(0.5, 0.5, cost=2.0)  # identical rows, twice the cost
    for order in ((a, b), (b, a)):
        chosen, table = acq.select_batch(jax.random.key(0), structs, data, list(order), XS, eps, 1, 100.0)
        assert order[chosen[0]].cost_mean == 1.0
        g = np.asarray(table.gain)
        np.testing.assert_allclose(g[0], g[1], rtol=1e-9)  # one draw, identity: the gain is deterministic
        # eq 11 of 2007.13553: argmax (H_n - J_n) / C, here with the hand-computed gain
        det = deterministic_gain(structs, data, a, XS, eps)
        np.testing.assert_allclose(g, det, rtol=1e-7)
        np.testing.assert_allclose(
            np.asarray(table.ratio), det / np.array([c.cost_mean for c in order]), rtol=1e-7
        )


def test_batch_is_greedy_and_a_pending_run_at_the_same_site_devalues_its_twin():
    structs, data, eps = one_draw_world()
    a = cand(0.0, 0.5)
    a2 = cand(0.0, 0.5)  # same site as a: the value is at the edge of Sigma_N, where sigma_epi is largest
    b = cand(0.9, 0.5)
    cands = [a, a2, b]
    key = jax.random.key(0)
    _, t0 = acq.select_batch(key, structs, data, cands, XS, eps, 1, 100.0)
    _, tp = acq.select_batch(key, structs, data, cands, XS, eps, 1, 100.0, pending=(a,))
    g0, gp = np.asarray(t0.gain), np.asarray(tp.gain)
    assert g0[1] > g0[2]  # without the pending run, the twin is the better second pick
    assert gp[1] < g0[1] * 0.9  # making a pending removes most of the twin's value
    assert gp[1] < gp[2]  # ... so the batch moves elsewhere
    chosen, _ = acq.select_batch(key, structs, data, cands, XS, eps, 3, 100.0)
    assert chosen[0] in (0, 1) and chosen[1] == 2 and len(set(chosen)) == len(chosen)
    assert set(chosen) <= {0, 1, 2}


def test_batch_stops_when_nothing_has_value():
    structs, data, _ = one_draw_world()
    chosen, _ = acq.select_batch(jax.random.key(0), structs, data, [cand(0.5, 0.5)], XS, 1e6, 3, 100.0)
    assert chosen == []  # hinge H_n = 0: P1 holds, no candidate has value


def test_select_batch_subsamples_draws_by_weight_deterministically():
    structs, data = world([make_params(24, c0=c) for c in np.linspace(0.0, 1.0, 6)])
    eps = 0.1
    cands = [cand(0.3, 0.5), cand(0.7, 0.5)]
    r1 = acq.select_batch(jax.random.key(0), structs, data, cands, XS, eps, 1, 100.0, max_draws=3)
    r2 = acq.select_batch(jax.random.key(0), structs, data, cands, XS, eps, 1, 100.0, max_draws=3)
    assert r1[0] == r2[0]
    np.testing.assert_allclose(r1[1].gain, r2[1].gain, rtol=1e-12)
