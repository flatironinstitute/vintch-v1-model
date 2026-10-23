"""Regularizers for the subunit model, built on NeMoS ``Regularizer``."""

from typing import Any

import jax
import jax.numpy as jnp
from nemos import tree_utils
from nemos.regularizer import Regularizer


class DifferencePenalty(Regularizer):
    r"""Smoothness penalty on differences, with an optional ridge term.

    For each parameter leaf :math:`\theta`,

    .. math::
        P(\theta) = \frac{\alpha}{2} \left( (1 - r) \lVert D_n \theta \rVert^2
        + r \lVert \theta \rVert^2 \right),

    where :math:`D_n` takes the ``order``-th differences, :math:`\alpha` is the strength
    and :math:`r` the ridge ratio. Differences never cross from one leaf (one channel)
    to the next. With :math:`r = 0`, polynomials of degree below ``order`` are not
    penalized; a small :math:`r > 0` penalizes them too.

    The strength follows ``ElasticNet``: a tuple ``(alpha, r)`` with ``0 <= r < 1``, or a
    scalar ``alpha`` for ``r = 0``.

    Parameters
    ----------
    order :
        Order of the differences: 1 penalizes slopes, 2 penalizes curvature.

    Notes
    -----
    Only scalar ``alpha`` and ``r`` are accepted: a value per coefficient does not fit
    the differences, which have ``order`` fewer entries than the leaf.

    No Hessian tag is declared (as for ``ElasticNet``): the base class expects one claim
    per strength leaf, which does not fit the ``(alpha, r)`` pairs.
    """

    _allowed_solvers = ("GradientDescent", "BFGS", "LBFGS", "NonlinearCG", "Newton")
    _default_solver = "Newton"

    def __init__(self, order: int = 2):
        self.order = order

    def _validate_strength(self, strength: Any):
        # Called by BaseRegressor's strength setter, and below for penalty_fn / penalized_loss.
        if strength is None:
            alpha, ratio = 1.0, 0.0
        elif isinstance(strength, tuple):
            if len(strength) != 2:
                raise TypeError("DifferencePenalty strength must be a tuple (alpha, ratio).")
            alpha, ratio = strength
        else:
            alpha, ratio = strength, 0.0
        alpha = super()._validate_strength(alpha)
        ratio = self._convert_strength(ratio)
        if jnp.ndim(alpha) != 0 or jnp.ndim(ratio) != 0:
            raise ValueError("DifferencePenalty takes a scalar alpha and a scalar ratio.")
        if not 0 <= ratio < 1:
            raise ValueError(f"DifferencePenalty ridge ratio must be in [0, 1), got {ratio}.")
        return alpha, ratio

    def _validate_strength_structure(self, params: Any, strength: Any):
        alpha, ratio = self._validate_strength(strength)
        alpha = super()._validate_strength_structure(params, alpha)
        ratio = super()._validate_strength_structure(params, ratio)
        return jax.tree.map(
            lambda a, r: None if a is None else (a, r), alpha, ratio, is_leaf=lambda x: x is None
        )

    def _penalty_on_subtree(self, subtree, strength: Any, **kwargs) -> jnp.ndarray:
        def penalty(coef, leaf_strength):
            alpha, ratio = leaf_strength
            smoothness = jnp.sum(jnp.square(jnp.diff(coef, n=self.order)))
            return 0.5 * alpha * ((1 - ratio) * smoothness + ratio * jnp.sum(jnp.square(coef)))

        return tree_utils.pytree_map_and_reduce(penalty, sum, subtree, strength)
