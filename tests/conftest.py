"""Test configuration: a persistent JAX compilation cache, so repeated runs do not recompile.

The cache directory is GCBML_JAX_CACHE if set, else ~/.cache/gcbml-jax. Compiled programs that took
longer than 0.5 s to compile are cached across processes (and across worktrees that share the cache).
"""

import os
from pathlib import Path

import jax

_cache = Path(os.environ.get("GCBML_JAX_CACHE", Path.home() / ".cache" / "gcbml-jax"))
_cache.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
