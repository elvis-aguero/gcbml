# gcbml — Grid-Convergent Bayesian Machine Learning

gcbml predicts the **converged** answer of a simulation code, f(x) at resolution h → 0, over a design region. It learns from runs on several grids, coarse and fine, and decides which run to buy next within a compute budget.

It extends the multi-fidelity KRR-LR-GPR framework of Yi et al. (2024, arXiv 2407.15110):
- every grid is a fidelity, linked to the converged value by a transfer that becomes exact as h → 0;
- the convergence order is learned with its uncertainty;
- inference is fully Bayesian (MCMC);
- the next run (condition and grid) maximises the expected uncertainty reduction per unit cost.

The method is specified in [docs/spec.md](docs/spec.md). Status: under construction (v0.1).

gcbml never runs simulations. You provide an *oracle* (see `gcbml.oracle`) that submits runs and reports results and costs.

## Install
    uv sync              # CPU
    uv sync --extra cuda # NVIDIA GPU
