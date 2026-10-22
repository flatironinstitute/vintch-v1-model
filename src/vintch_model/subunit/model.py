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
import nemos as nmo
from jaxtyping import Array, Float, PyTree
from nemos.observation_models import Observations

from .params import SubunitParams


class BasisNonlinearity(eqx.Module):
    """Nonlinearity f(s) = sum_l coef[l] * B_l(u), with u = (s - lo) / (hi - lo).

    The nonlinearity is built from a one-dimensional NeMoS basis defined on the
    interval (0, 1). Before the basis is evaluated, each pre-activation ``s`` is rescaled
    to ``u = (s - lo) / (hi - lo)``, where ``bounds = (lo, hi)`` is passed to each call.
    This way the basis covers pre-activations from ``lo`` to ``hi``.

    Parameters
    ----------
    basis :
        A NeMoS basis with bounds (0, 1), e.g. ``BSplineEval(order=2)`` for tents.
    fill_strategy :
        Behavior for pre-activations outside ``bounds``: ``"clamp"`` evaluates them at the
        nearest end; ``"zero"`` gives the basis' ``fill_value`` (0 for ``tents``).
        Note that "clamp" is the original choice in Vintch 2015.

    Notes
    -----
    The basis must have its bounds fixed to (0, 1). Without fixed bounds, NeMoS
    rescales each input to its own range, so the same input would give different
    values on different data. The rescaling to ``(lo, hi)`` is done here instead, with
    ``bounds`` passed to each call: the bounds move during a fit, and the basis is a
    static field, so changing its bounds would trigger recompilation.
    """

    basis: Any = eqx.field(static=True)
    fill_strategy: Literal["clamp", "zero"] = eqx.field(static=True, default="clamp")

    def _features_leaf(
        self, pre_activations: Float[Array, "..."], bounds: Float[Array, "2"]
    ) -> Float[Array, "... n_basis"]:
        """Apply the basis to an array leaf."""
        lo, hi = bounds
        u = (pre_activations - lo) / (hi - lo)
        if self.fill_strategy == "clamp":
            # Largest value below 1 in u's dtype: in float32, the jitted nemos B-spline
            # gives all zeros at an input exactly on the upper bound.
            u = jnp.clip(u, 0.0, jnp.nextafter(jnp.ones((), u.dtype), 0))
        return self.basis.evaluate(u)

    def features(
        self, pre_activations: PyTree[Float[Array, "..."]], bounds: PyTree[Float[Array, "2"]]
    ) -> PyTree[Float[Array, "... n_basis"]]:
        """Basis functions evaluated at each pre-activation, per channel."""
        return jax.tree.map(self._features_leaf, pre_activations, bounds)

    def __call__(
        self,
        coef: PyTree[Float[Array, " n_basis"]],
        pre_activations: PyTree[Float[Array, "..."]],
        bounds: PyTree[Float[Array, "2"]],
    ) -> PyTree[Float[Array, "..."]]:
        if self.basis.bounds != (0, 1):
            raise RuntimeError("Misconfigured basis bounds. To work as a BasisNonlinearity, "
                               f"a NeMoS basis must have bounds set to (0,1).\n"
                               f"Provided bounds: {self.basis.bounds}")
        return jax.tree.map(
            lambda c, z, b: self._features_leaf(z, b) @ c, coef, pre_activations, bounds
        )


class NoNormalization(eqx.Module):
    """Kernel used as it is (the original Vintch et al. 2015 fit)."""

    def __call__(
        self, kernel: PyTree[Float[Array, "lags k_rows k_cols"]]
    ) -> PyTree[Float[Array, "lags k_rows k_cols"]]:
        return kernel


class L2Normalization(eqx.Module):
    """Each channel's kernel divided by its L2 norm, so only its shape is a parameter.

    ``params.kernel`` stays the raw parameter; the normalized kernel exists only inside
    the forward pass.
    """

    def __call__(
        self, kernel: PyTree[Float[Array, "lags k_rows k_cols"]]
    ) -> PyTree[Float[Array, "lags k_rows k_cols"]]:
        return jax.tree.map(lambda k: k / jnp.linalg.norm(k), kernel)


class Convolution(eqx.Module):
    """Kernel applied at every spatial position (cross-correlation, no kernel flip).

    The lags are the input channels of a 2-D convolution: the kernel spans all lags and
    slides over space only. With ``padding="VALID"`` and unit strides,
    ``pre_activations[t, i, j] = sum(kernel * stimulus[t, :, i:i + k_rows, j:j + k_cols])``,
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
        kernel: PyTree[Float[Array, "lags k_rows k_cols"]],
        stimulus: Float[Array, "time lags rows cols"],
    ) -> PyTree[Float[Array, "time p_rows p_cols"]]:
        def conv(k):
            out = jax.lax.conv_general_dilated(stimulus, k[None], self.strides, self.padding)
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
        return jax.tree.map(lambda w, a: jnp.einsum("tij,ij->t", a, w), weights, activations)


class SubunitModel(eqx.Module):
    """Generator signal: offset + sum over channels of pooled subunit activations.

    Configuration only; parameters come in a ``SubunitParams``. The three stages are
    exposed separately, so a fit that holds the earlier stages fixed can compute them
    once and start from the stage it changes.
    """

    kernel_normalization: NoNormalization | L2Normalization
    convolution: Convolution
    nonlinearity: BasisNonlinearity
    pooling: Pooling

    def pre_activations(
        self, params: SubunitParams, stimulus: Float[Array, "time lags rows cols"]
    ) -> PyTree[Float[Array, "time p_rows p_cols"]]:
        """Normalized kernel applied at every position (uses ``params.kernel``)."""
        return self.convolution(self.kernel_normalization(params.kernel), stimulus)

    def activations(
        self,
        params: SubunitParams,
        pre_activations: PyTree[Float[Array, "time p_rows p_cols"]],
        bounds: PyTree[Float[Array, "2"]],
    ) -> PyTree[Float[Array, "time p_rows p_cols"]]:
        """Subunit nonlinearity (uses ``params.coef``)."""
        return self.nonlinearity(params.coef, pre_activations, bounds)

    def output(
        self, params: SubunitParams, activations: PyTree[Float[Array, "time p_rows p_cols"]]
    ) -> Float[Array, " time"]:
        """Pooling and sum over channels (uses ``params.pooling_weights`` and ``params.offset``)."""
        pooled = self.pooling(params.pooling_weights, activations)
        return params.offset + jax.tree.reduce(jnp.add, pooled)

    def __call__(
        self,
        params: SubunitParams,
        stimulus: Float[Array, "time lags rows cols"],
        bounds: PyTree[Float[Array, "2"]],
    ) -> Float[Array, " time"]:
        pre_activations = self.pre_activations(params, stimulus)
        activations = self.activations(params, pre_activations, bounds)
        return self.output(params, activations)


class Loss(eqx.Module):
    """Negative log-likelihood of the observations under the model's prediction.

    The call signature ``(params, *args)`` is the one NeMoS solvers expect for the
    unregularized loss.

    Parameters
    ----------
    model :
        The subunit model.
    observation_model :
        A NeMoS observation model, e.g. ``GaussianObservations()``. With the Gaussian
        model the loss is the mean squared error.
    """

    model: SubunitModel
    observation_model: Observations = eqx.field(static=True)

    def __call__(
        self,
        params: SubunitParams,
        stimulus: Float[Array, "time lags rows cols"],
        y: Float[Array, " time"],
        bounds: PyTree[Float[Array, "2"]],
    ) -> Float[Array, ""]:
        prediction = self.model(params, stimulus, bounds)
        return self.observation_model._negative_log_likelihood(y, prediction)


def tents(
    n_basis_funcs: int, fill_strategy: Literal["clamp", "zero"] = "clamp"
) -> BasisNonlinearity:
    """Tent (piecewise-linear) nonlinearity: nodes at ``linspace(*bounds, n_basis_funcs)``.

    Safe constructor of a tent nonlinearity. The constructor enforces bounds and fill value.
    """
    basis = nmo.basis.BSplineEval(
        n_basis_funcs=n_basis_funcs, order=2, bounds=(0.0, 1.0), fill_value=0.0
    )
    return BasisNonlinearity(basis, fill_strategy)
