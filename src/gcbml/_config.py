"""Package-wide numerical configuration.

Importing gcbml turns on 64-bit floats in JAX. Every covariance computation in this package
needs float64: Cholesky factors of kernels with h^{2p} scaling lose all accuracy in float32.

Shapes that change with the number of data n would trigger a JAX recompilation on every new
run. Data arrays are therefore padded to a bucket size (``bucket(n)``) with masked rows.
"""

from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

BUCKET_GROWTH = 1.25
BUCKET_MIN = 16


def bucket(n: int) -> int:
    """Smallest bucket size >= n. Buckets are BUCKET_MIN * BUCKET_GROWTH**k, rounded up to a multiple of 8."""
    if n <= BUCKET_MIN:
        return BUCKET_MIN
    k = math.ceil(math.log(n / BUCKET_MIN) / math.log(BUCKET_GROWTH))
    size = BUCKET_MIN * BUCKET_GROWTH**k
    return int(8 * math.ceil(size / 8))
