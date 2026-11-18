"""Components of the subunit model.

Conventions:

- Parameters are the first argument of every call.
- Inputs that are not learned, such as the stimulus or the nonlinearity's bounds,
  come after the parameters.
- Channels are the structure of a pytree. A kernel tree such as
  ``{"excitatory": k1, "suppressive": k2}`` gives pre-activations with the same structure,
  and the nonlinearity's ``coef`` and ``bounds`` are trees with that structure too.

NeMoS is used wherever it fits: the tent nonlinearity is an order-2 B-spline basis,
and any other NeMoS basis can be wrapped the same way.
"""

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .basis import BasisNonlinearity
from .params import SubunitParams


class Convolution(eqx.Module):
    """Kernel applied at every spatial position (cross-correlation, no kernel flip).

    The lags are the input channels of a 2-D convolution: the kernel spans all lags and
    slides over space only. With ``padding="VALID"`` and unit strides,
    ``output[t, i, j] = sum(kernel * stimulus[t, :, i:i + k_rows, j:j + k_cols])``,
    where ``k_rows`` and ``k_cols`` are the kernel's spatial size. ``kernel`` is a tree
    with one kernel per channel; the pre-activations have the same structure.

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
        kernel: PyTree[Float[Array, "lags k_height k_width"]],
        stimulus: Float[Array, "time lags rows cols"],
    ) -> PyTree[Float[Array, "time p_height p_width"]]:
        def conv(k):
            out = jax.lax.conv_general_dilated(
                stimulus, k[None], self.strides, self.padding
            )
            return out[:, 0]

        return jax.tree.map(conv, kernel)


class Pooling(eqx.Module):
    """Weighted sum of subunit activations over positions, per channel.

    ``weights`` and ``activations`` are trees of the same structure (one leaf per channel);
    the result has that structure, with one signal per channel.
    """

    def __call__(
        self,
        weights: PyTree[Float[Array, "p_rows p_cols"]],
        activations: PyTree[Float[Array, "time p_rows p_cols"]],
    ) -> PyTree[Float[Array, " time"]]:
        return jax.tree.map(
            lambda w, a: jnp.einsum("ij,tij->t", w, a), weights, activations
        )


class ReduceAndOffset(eqx.Module):
    def __call__(
        self, offset: Float[Array, ""], pooled: PyTree[Float[Array, " time"]]
    ) -> Float[Array, " time"]:
        """Reduce the pooled activations, and apply an offset."""
        return offset + jax.tree.reduce(jnp.add, pooled)


class SubunitModel(eqx.Module):
    """Generator signal: offset + sum over channels of pooled subunit activations.

    Configuration only; parameters come in a ``SubunitParams``. The three stages are
    exposed separately, so a fit that holds the earlier stages fixed can compute them
    once and start from the stage it changes.
    """

    convolution: Convolution
    nonlinearity: BasisNonlinearity
    pooling: Pooling
    reducing: ReduceAndOffset

    def pre_activations(
        self,
        kernel: PyTree[Float[Array, "lags k_rows k_cols"]],
        stimulus: Float[Array, "time lags rows cols"],
    ) -> PyTree[Float[Array, "time p_rows p_cols"]]:
        """Kernel applied at every position (uses ``params.kernel``)."""
        return self.convolution(kernel, stimulus)

    def activations(
        self,
        coef: PyTree[Float[Array, " n_basis"]],
        pre_activations: PyTree[Float[Array, "time p_rows p_cols"]],
        bounds: PyTree[Float[Array, "2"]],
    ) -> PyTree[Float[Array, "time p_rows p_cols"]]:
        """Subunit nonlinearity (uses ``params.coef``)."""
        return self.nonlinearity(coef, pre_activations, bounds)

    def pool(
        self,
        pooling_weights: PyTree[Float[Array, "p_rows p_cols"]],
        activations: PyTree[Float[Array, "time p_rows p_cols"]],
    ) -> PyTree[Float[Array, " time"]]:
        return self.pooling(pooling_weights, activations)

    def reduce(
        self, offset: Float[Array, ""], pooled: PyTree[Float[Array, " time"]]
    ) -> Float[Array, " time"]:
        return self.reducing(offset, pooled)

    def __call__(
        self,
        params: SubunitParams,
        stimulus: Float[Array, "time lags rows cols"],
        bounds: PyTree[Float[Array, "2"]],
    ) -> Float[Array, " time"]:
        pre_activations = self.pre_activations(params.kernel, stimulus)
        activations = self.activations(params.coef, pre_activations, bounds)
        pooled = self.pool(params.pooling_weights, activations)
        return self.reduce(params.offset, pooled)
