"""Components of the subunit model.

Components hold configuration only. Parameters are passed as the first argument of
every call; data that is fixed during a solve (such as the nonlinearity's bounds)
follows as further arguments.

Channels are the structure of a pytree: a kernel tree such as
``{"excitatory": k1, "suppressive": k2}`` gives drives with the same structure, and
the nonlinearity's ``coef`` and ``bounds`` are trees of that structure too.
"""

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import nemos as nmo
from jaxtyping import Array, Float, PyTree

from .params import SubunitParams


class BasisNonlinearity(eqx.Module):
    """Nonlinearity f(s) = sum_l coef[l] * B_l(u), u = (s - lo) / (hi - lo).

    The basis lives on fixed bounds (0, 1); ``bounds = (lo, hi)`` places it on the drive
    axis. ``bounds`` is data, not configuration, so it can be re-placed between solves
    without changing this object. Every channel uses the same basis; ``coef``, ``drive``
    and ``bounds`` are trees of the same structure (one leaf per channel, each ``bounds``
    leaf a length-2 array).

    Parameters
    ----------
    basis :
        A nemos basis evaluated on (0, 1), e.g. ``BSplineEval(order=2)`` for tents.
    fill_strategy :
        Behavior for drives outside ``bounds``: ``"clamp"`` evaluates them at the
        nearest end; ``"zero"`` gives the basis' ``fill_value`` (0 for ``tents``).
    """

    basis: Any = eqx.field(static=True)
    fill_strategy: Literal["clamp", "zero"] = eqx.field(static=True, default="clamp")

    def _features_leaf(
        self, drive: Float[Array, "..."], bounds: Float[Array, "2"]
    ) -> Float[Array, "... n_basis"]:
        lo, hi = bounds
        u = (drive - lo) / (hi - lo)
        if self.fill_strategy == "clamp":
            # Largest value below 1 in u's dtype: in float32, the jitted nemos B-spline
            # gives all zeros at an input exactly on the upper bound.
            u = jnp.clip(u, 0.0, jnp.nextafter(jnp.ones((), u.dtype), 0))
        return self.basis.evaluate(u)

    def features(
        self, drive: PyTree[Float[Array, "..."]], bounds: PyTree[Float[Array, "2"]]
    ) -> PyTree[Float[Array, "... n_basis"]]:
        """Basis functions evaluated at each drive value, per channel."""
        return jax.tree.map(self._features_leaf, drive, bounds)

    def __call__(
        self,
        coef: PyTree[Float[Array, " n_basis"]],
        drive: PyTree[Float[Array, "..."]],
        bounds: PyTree[Float[Array, "2"]],
    ) -> PyTree[Float[Array, "..."]]:
        return jax.tree.map(lambda c, d, b: self._features_leaf(d, b) @ c, coef, drive, bounds)


class Convolution(eqx.Module):
    """Kernel applied at every spatial position (cross-correlation, no kernel flip).

    The lags are the input channels of a 2-D convolution: the kernel spans all lags and
    slides over space only, so ``drive[t, i, j] = sum(kernel * stimulus[t, :, i:i+kr, j:j+kc])``
    for ``padding="VALID"`` and unit strides. ``kernel`` is a tree with one kernel per
    channel; the drive has the same structure.

    Parameters
    ----------
    padding :
        Padding passed to ``jax.lax.conv_general_dilated`` (e.g. ``"VALID"``, ``"SAME"``).
    strides :
        Spatial strides.
    """

    padding: str = eqx.field(static=True, default="VALID")
    strides: tuple[int, int] = eqx.field(static=True, default=(1, 1))

    def __call__(
        self,
        kernel: PyTree[Float[Array, "lags k_rows k_cols"]],
        stimulus: Float[Array, "time lags rows cols"],
    ) -> PyTree[Float[Array, "time p_rows p_cols"]]:
        def conv(k):
            out = jax.lax.conv_general_dilated(stimulus, k[None], self.strides, self.padding)
            return out[:, 0]

        return jax.tree.map(conv, kernel)


class Pooling(eqx.Module):
    """Weighted sum of subunit responses over positions, per channel.

    ``weights`` and ``responses`` are trees of the same structure (one leaf per channel);
    the result has that structure, with one signal per channel.
    """

    def __call__(
        self,
        weights: PyTree[Float[Array, "p_rows p_cols"]],
        responses: PyTree[Float[Array, "time p_rows p_cols"]],
    ) -> PyTree[Float[Array, " time"]]:
        return jax.tree.map(lambda w, r: jnp.einsum("tij,ij->t", r, w), weights, responses)


class SubunitModel(eqx.Module):
    """Generator signal: intercept + sum over channels of pooled subunit responses.

    Configuration only; parameters come in a ``SubunitParams``.
    """

    convolution: Convolution
    nonlinearity: BasisNonlinearity
    pooling: Pooling

    def __call__(
        self,
        params: SubunitParams,
        stimulus: Float[Array, "time lags rows cols"],
        bounds: PyTree[Float[Array, "2"]],
    ) -> Float[Array, " time"]:
        drive = self.convolution(params.kernel, stimulus)
        responses = self.nonlinearity(params.coef, drive, bounds)
        pooled = self.pooling(params.pooling_weights, responses)
        return params.intercept + jax.tree.reduce(jnp.add, pooled)


def tents(
    n_basis_funcs: int, fill_strategy: Literal["clamp", "zero"] = "clamp"
) -> BasisNonlinearity:
    """Tent (piecewise-linear) nonlinearity: nodes at ``linspace(*bounds, n_basis_funcs)``."""
    basis = nmo.basis.BSplineEval(
        n_basis_funcs=n_basis_funcs, order=2, bounds=(0.0, 1.0), fill_value=0.0
    )
    return BasisNonlinearity(basis, fill_strategy)
