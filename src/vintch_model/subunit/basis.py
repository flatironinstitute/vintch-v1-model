"""Helper functions for subunit model."""

from __future__ import annotations

from typing import Any, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import nemos as nmo
from jaxtyping import Array, Float, PyTree


class BasisNonlinearity(eqx.Module):
    """Nonlinearity f(s) = sum_l coef[l] * B_l(u), with u = (s - lo) / (hi - lo).

    The nonlinearity is built from a one-dimensional NeMoS basis defined on the
    interval (0, 1). Before the basis is evaluated, each pre-activation ``s`` is rescaled
    to ``u = (s - lo) / (hi - lo)``, where ``bounds = (lo, hi)`` is passed to each call.
    This way the basis covers pre-activations from ``lo`` to ``hi``.

    Parameters
    ----------
    basis :
        Either the number of basis as an int, or a NeMoS basis with bounds (0, 1),
        e.g. ``BSplineEval(order=2)`` for tents.
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

    # still needed to mark field as static
    basis: Any = eqx.field(static=True)
    fill_strategy: Literal["clamp", "zero"] = eqx.field(static=True)

    def __init__(
        self,
        basis: Optional[Any | int] = 8,
        fill_strategy: Literal["zero", "clamp"] = "clamp",
    ):
        bounds = getattr(basis, "bounds", (0, 1))
        if bounds != (0, 1):
            raise ValueError(
                "Misconfigured basis bounds. To work as a BasisNonlinearity, "
                f"a NeMoS basis must have bounds set to (0,1).\n"
                f"Basis has {bounds=}"
            )

        if isinstance(basis, int):
            self.basis = tents(basis, fill_strategy=fill_strategy)
        else:
            if not hasattr(basis, "evaluate"):
                raise ValueError("A valid basis needs an ``evaluate`` method.")
            self.basis = basis

        self.fill_strategy = fill_strategy

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

    def __call__(
        self,
        coef: PyTree[Float[Array, " n_basis"]],
        pre_activations: PyTree[Float[Array, "..."]],
        bounds: PyTree[Float[Array, "2"]],
    ) -> PyTree[Float[Array, "..."]]:
        return jax.tree.map(
            lambda c, z, b: self._features_leaf(z, b) @ c, coef, pre_activations, bounds
        )


def tents(
    n_basis_funcs: int, fill_strategy: Literal["clamp", "zero"] = "clamp"
) -> BasisNonlinearity:
    """Tent (piecewise-linear) nonlinearity: nodes at ``linspace(*bounds, n_basis_funcs)``.

    Safe constructor of a tent nonlinearity. The constructor enforces bounds and fill value.

    Parameters
    ----------
    n_basis_funcs:
        The number of basis functions to use for the tent nonlinearity.
    fill_strategy:
        The strategy to fill the tent nonlinearity, `"zero"` fills with zero, `"clamp"`
        clamps the out-of-bounds values.

    Raises
    ------
    ValueError
        - If ``n_basis_funcs`` is not a strictly positive integer.
        - If ``fill_strategy`` is neither ``"clamp"`` nor ``"zero"``.

    """
    int_n_basis = int(n_basis_funcs)
    if int_n_basis <= 0:
        raise ValueError("``n_basis_funcs`` must be strictly positive.")
    elif int_n_basis != n_basis_funcs:
        raise ValueError("``n_basis_funcs`` must be an integer.")

    if fill_strategy not in ["zero", "clamp"]:
        raise ValueError("fill_strategy must be 'zero' or 'clamp'.")

    basis = nmo.basis.BSplineEval(
        n_basis_funcs=n_basis_funcs, order=2, bounds=(0.0, 1.0), fill_value=0.0
    )
    return BasisNonlinearity(basis, fill_strategy)
