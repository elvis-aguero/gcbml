# Contributing to gcbml

## Rules
1. **Test first.** Write the test, run it, and see it fail for the right reason. Then write the code. A test that passes on the first run proves nothing.
2. **Known answers.** Every numerical test compares against an independent reference: a closed form, a brute-force NumPy computation, or a published value. "It runs" is not a test.
3. **No explicit inverses.** Use `gcbml.linalg` (Cholesky, triangular solves). `np.linalg.inv` and `jnp.linalg.inv` are not allowed.
4. **float64 always.** `import gcbml` enables it. Never cast to float32.
5. **Pure and jittable.** Numerical functions take arrays and return arrays, with no Python side effects, so that `jax.jit` and `jax.vmap` work. Data-dependent Python control flow is not allowed inside jitted code; use `lax.cond`, `lax.while_loop` and `lax.scan`.
6. **Cite what you implement.** Every docstring that implements a published formula names the arXiv id and the equation or section. Check the number in the paper (the arXiv MCP tools can read it). Never write a number from memory.
7. **No application content.** gcbml knows nothing about any simulator. Physical prior scales are user inputs.
8. **Interfaces are fixed.** Do not change a signature in a stub without the reviewer's approval. If an interface is wrong, say so in your report.

## Commands
- Tests: `uv run pytest` (fast). `uv run pytest -m slow` for the statistical and benchmark tests (run them on a compute node).
- Lint: `uv run ruff check src tests` and `uv run ruff format --check src tests`.

## Style
- Line length 110. Type hints on public functions. Docstrings are plain prose with the math inline.
- Test names say the behaviour: `test_lb_with_gamma_half_equals_brownian`.
- Plain `assert`, `np.random.default_rng(seed)` or `jax.random.key(seed)`, and no test classes.
