"""NUTS on the global block theta inside the Gibbs scheme of inference.py (experimental, not the default).

One iteration, for each of 4 chains:
  1. theta (log sigma_mu, log ell_mu, c0, c1, log p0, log sigma_delta, log ell_x, log ell_h or logit gamma,
     m_s, b_s): one NUTS transition (blackjax 1.x, multinomial NUTS with iterative trajectory expansion) on
     log p(z | theta, zeta) + log prior(theta), with zeta and (log sigma_z, log ell_z) held fixed. The
     gradient comes from JAX through model.log_marginal (needs the custom JVP of linalg and the safe sqrt of
     kernels.ard_matern).
  2. zeta: elliptical slice, 3. (log sigma_z, log ell_z): surrogate-data slice sweep, 3b. the whitened move
     (all three as in inference._Sampler.make_step, same functions).
Warm-up: Stan's windowed adaptation (blackjax.adaptation.window_adaptation.base, schedule from
build_schedule) of the step size (dual averaging, target acceptance 0.8) and a diagonal inverse mass matrix,
computed on the conditional of theta given the current (zeta, hz). The slice widths of hz are set from the
warm-up draws as in mcmc.chains.adapt_widths. Starting points: the rule of inference._Sampler.init_state.
Cost accounting: info["grad_evals"] counts NUTS leapfrog gradient evaluations (+ 1 per iteration for the
state initialisation); info["lik_evals"] counts the likelihood evaluations of the slice and elliptical moves.

fit_nuts(key, data, z, bounds, cfg, scales, n_controls, n_warmup, n_samples, n_chains=4, max_doublings=8,
         target_accept=0.8) -> NutsPosterior   (the fields of inference.Posterior, plus info)
Censored outputs and varying order are not supported.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.window_adaptation import base as _window_base
from blackjax.adaptation.window_adaptation import build_schedule

from gcbml import inference, linalg
from gcbml.mcmc import diagnostics as dg
from gcbml.mcmc.chains import WIDTH_FLOOR, chain_map
from gcbml.mcmc.elliptical import ess_step
from gcbml.mcmc.surrogate import surrogate_slice_step

INIT_STEP_SIZE = 0.05
_CACHE: dict = {}  # compiled chain programs; each closes over the template of its first call


class NutsPosterior(NamedTuple):
    params: Any
    z: Any
    theta: dict
    diagnostics: dict
    n_evals: Any
    info: dict  # per chain: grad_evals, lik_evals, divergences, mean_accept, mean_steps, step_size


def _make_chain(template, n_warmup: int, n_samples: int, max_doublings: int, target_accept: float):
    kernel = blackjax.nuts.build_kernel()
    schedule = jnp.asarray(build_schedule(n_warmup), dtype=jnp.int32) if n_warmup > 0 else None

    def chain(key, aux):
        smp = template.with_aux(aux)
        pi = smp._no_pi()
        k_init, k_warm, k_samp = jax.random.split(key, 3)
        st0, _ = smp.init_state(k_init)

        def one(key, st, step_size, imm, w_hz):
            ks = jax.random.split(key, 4)
            theta, hz, zeta, z = st["theta"], st["hz"], st["zeta"], st["z"]

            def ld(v):
                return smp._theta_logdensity(v, hz, zeta, pi, z)

            ns = blackjax.nuts.init(theta, ld)
            ns, info = kernel(ks[0], ns, ld, step_size, imm, max_doublings)
            theta = ns.position
            t = smp.layout.unpack(theta)
            L = linalg.factor(smp.zeta_cov(hz), aux.site_mask).L
            zeta, n2 = ess_step(ks[1], zeta, L, lambda f: smp.loglik(t, f, pi, z))
            hz, zeta, n3 = surrogate_slice_step(
                ks[2], hz, zeta, smp.zeta_cov, lambda f: smp.loglik(t, f, pi, z), smp.log_prior_hz,
                smp.zeta_surrogate_var, w_hz,
            )  # fmt: skip
            hz, zeta, n3b = smp._whitened_hz_move(ks[3], hz, zeta, w_hz, lambda f: smp.loglik(t, f, pi, z))
            out = {"theta": theta, "hz": hz, "zeta": zeta, "z": z}
            stats = {
                "grad_evals": info.num_integration_steps + 1,
                "lik_evals": n2 + n3 + n3b,
                "div": info.is_divergent.astype(jnp.int32),
                "accept": info.acceptance_rate,
            }
            return out, stats

        w0 = jnp.ones(smp.n_hz)
        if n_warmup > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                ad_init, ad_update, ad_final = _window_base(True, target_accept)
            ad0 = ad_init(st0["theta"], INIT_STEP_SIZE)

            def warm(carry, xs):
                st, ad = carry
                k, stage = xs
                st, stats = one(k, st, ad.step_size, ad.inverse_mass_matrix, w0)
                ad = ad_update(ad, (stage[0], stage[1].astype(bool)), st["theta"], stats["accept"])
                return (st, ad), (st["hz"], stats)

            (st1, ad1), (hz_hist, wstats) = jax.lax.scan(
                warm, (st0, ad0), (jax.random.split(k_warm, n_warmup), schedule)
            )
            step_size, imm = ad_final(ad1)
            second = hz_hist[n_warmup // 2 :]
            w_hz = jnp.maximum(2.0 * jnp.std(second, axis=0, ddof=1), WIDTH_FLOOR) if n_warmup >= 4 else w0
        else:
            st1, step_size, imm, w_hz = st0, jnp.asarray(INIT_STEP_SIZE), jnp.ones_like(st0["theta"]), w0
            wstats = None

        def samp(st, k):
            st, stats = one(k, st, step_size, imm, w_hz)
            return st, (st, stats)

        _, (draws, sstats) = jax.lax.scan(samp, st1, jax.random.split(k_samp, n_samples))
        params = jax.vmap(smp.constrain)(draws)
        info = {
            "grad_evals": jnp.sum(sstats["grad_evals"]),
            "lik_evals": jnp.sum(sstats["lik_evals"]),
            "divergences": jnp.sum(sstats["div"]),
            "mean_accept": jnp.mean(sstats["accept"]),
            "mean_steps": jnp.mean(sstats["grad_evals"]),
            "step_size": step_size,
            "warm_grad_evals": jnp.sum(wstats["grad_evals"]) if wstats is not None else jnp.int32(0),
            "warm_lik_evals": jnp.sum(wstats["lik_evals"]) if wstats is not None else jnp.int32(0),
        }
        return draws, params, info

    return chain


def fit_nuts(
    key,
    data,
    z,
    bounds,
    cfg,
    scales,
    n_controls: int,
    n_warmup: int,
    n_samples: int,
    n_chains: int = 4,
    max_doublings: int = 8,
    target_accept: float = 0.8,
) -> NutsPosterior:
    """NUTS-in-Gibbs posterior (module docstring)."""
    if bool(np.any(np.asarray(data.censored, dtype=bool) & np.asarray(data.mask, dtype=bool))):
        raise NotImplementedError("fit_nuts does not support censored outputs")
    sampler = inference._Sampler(data, z, bounds, cfg, scales, n_controls, False)
    ck = (inference._template_key(sampler), tuple(a.shape for a in sampler.aux))
    ck += (int(n_warmup), int(n_samples), int(max_doublings), float(target_accept), int(n_chains))
    fn = _CACHE.get(ck)
    if fn is None:
        chain = _make_chain(sampler, int(n_warmup), int(n_samples), int(max_doublings), float(target_accept))
        fn = _CACHE[ck] = chain_map(chain, n_chains, 1)
        while len(_CACHE) > 4:
            _CACHE.pop(next(iter(_CACHE)))
    draws, params, info = fn(jax.random.split(key, n_chains), sampler.aux)
    theta = sampler.layout.unpack(draws["theta"])
    theta["log_sigma_z"], theta["log_ell_z"] = draws["hz"][..., 0], draws["hz"][..., 1:]
    theta = {name: np.asarray(v) for name, v in theta.items()}
    scalars = {}
    for name, v in theta.items():
        flat = v.reshape(v.shape[:2] + (-1,))
        for i in range(flat.shape[-1]):
            scalars[name if flat.shape[-1] == 1 and v.ndim == 2 else f"{name}[{i}]"] = flat[:, :, i]
    info = {k: np.asarray(v) for k, v in info.items()}
    n_evals = int(
        info["grad_evals"].sum()
        + info["lik_evals"].sum()
        + info["warm_grad_evals"].sum()
        + info["warm_lik_evals"].sum()
    )
    return NutsPosterior(params, draws["z"], theta, dg.summary(scalars), n_evals, info)
