"""Fitting procedures for the subunit model.

Functions prefixed ``replica_`` belong to the original fit of Vintch et al. (2015)
and reproduce its choices; the others are the general fit.
"""

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from nemos.regularizer import Regularizer
from nemos.solvers import get_solver

from .params import SubunitParams


def fix_params_from(params: SubunitParams, learned: SubunitParams) -> SubunitParams:
    """NeMoS-style ``fix_params`` that holds every parameter not learned at its current value.

    Parameters
    ----------
    params :
        Current values of all parameters.
    learned :
        Prefix tree of booleans marking what is learned, e.g.
        ``SubunitParams(kernel=False, coef=False, pooling_weights=True, offset=True)``.

    Returns
    -------
    :
        ``params`` with ``None`` at every learned leaf.
    """
    return jax.tree.map(
        lambda flag, p: jax.tree.map(lambda _: None, p) if flag else p, learned, params
    )


def solve(
    loss: Callable,
    params: SubunitParams,
    fix_params: SubunitParams,
    regularizer: Regularizer,
    strength: Any,
    *args,
    solver_name: str | None = None,
    **solver_kwargs,
) -> SubunitParams:
    """Solve for the parameters that are ``None`` in ``fix_params``.

    ``fix_params`` follows the NeMoS convention and mirrors ``params`` down to the leaves:
    an array leaf holds that parameter at the given value, a ``None`` leaf marks it as
    learned. The solver sees only the learned leaves; the loss sees them combined with
    the fixed ones.

    Parameters
    ----------
    loss :
        Unregularized loss, called as ``loss(params, *args)``.
    params :
        Starting values of all parameters (the fixed leaves are ignored).
    fix_params :
        Which parameters are held fixed, and at what value.
    regularizer, strength :
        Penalty on the learned parameters, as for any NeMoS solver.
    *args :
        Passed to ``loss`` after the parameters.
    solver_name :
        A NeMoS solver name (e.g. ``"Newton"``, ``"LBFGS"``). Defaults to the
        regularizer's default solver; the regularizer checks that it is allowed.
    **solver_kwargs :
        Passed to the solver.

    Returns
    -------
    :
        All parameters: the solved ones combined with the fixed ones.
    """
    learned = jax.tree.map(lambda x: x is None, fix_params, is_leaf=lambda x: x is None)
    active, frozen = eqx.partition(eqx.combine(fix_params, params), learned)

    def active_loss(p, *a):
        return loss(eqx.combine(p, frozen), *a)

    if solver_name is None:
        solver_name = regularizer.default_solver

    regularizer.check_solver(solver_name)
    solver_cls = get_solver(solver_name).implementation
    solver = solver_cls(
        active_loss, regularizer, strength, has_aux=False, init_params=active, **solver_kwargs
    )
    active, _, _ = solver.run(active, *args)
    return eqx.combine(active, frozen)


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
