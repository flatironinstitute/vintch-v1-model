"""Parameter container of the subunit model."""

from typing import Callable

from jaxtyping import Array, Float, PyTree
from nemos.params import ModelParams


class SubunitParams(ModelParams):
    """Parameters of the subunit model, grouped by component.

    ``kernel``, ``coef`` and ``pooling_weights`` are trees of the same structure, one leaf per
    channel (e.g. ``{"excitatory": ..., "suppressive": ...}``).

    Parameters
    ----------
    kernel :
        Subunit kernel per channel, ``[lags, k_rows, k_cols]``.
    coef :
        Nonlinearity basis coefficients per channel, ``[n_basis]``.
    pooling_weights :
        Pooling map per channel, ``[p_rows, p_cols]``.
    intercept :
        Constant added to the sum of the channels.
    """

    kernel: PyTree[Float[Array, "lags k_rows k_cols"]]
    coef: PyTree[Float[Array, " n_basis"]]
    pooling_weights: PyTree[Float[Array, "p_rows p_cols"]]
    intercept: Float[Array, ""]

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["SubunitParams"], PyTree]]:
        """Pooling maps and nonlinearity coefficients; never the kernel or the intercept."""
        return [lambda p: p.pooling_weights, lambda p: p.coef]
