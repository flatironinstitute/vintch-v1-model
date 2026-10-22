"""Fitting procedures for the subunit model.

Functions prefixed ``replica_`` belong to the original fit of Vintch et al. (2015)
and reproduce its choices; the others are the general fit.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


def replica_quantile_bounds(
    pre_activations: PyTree[Float[Array, "time p_rows p_cols"]],
) -> PyTree[Float[Array, "2"]]:
    """Tent bounds of the original fit (Vintch et al., 2015).

    Each channel's bounds are the 1/32 and 31/32 quantiles of its pre-activations, over
    all frames and positions. The original recomputes them before every tent solve.
    """
    q = jnp.array([1 / 32, 31 / 32])
    # Close to, but not exactly, the original rule: MATLAB takes the sorted value at
    # 0-based index round(q * n) - 1, "nearest" takes index round(q * (n - 1)). The two
    # can pick neighboring order statistics; the difference in the bounds is negligible.
    return jax.tree.map(lambda z: jnp.quantile(z, q, method="nearest"), pre_activations)
