"""Helper functions for subunit model."""

from typing import Literal, TYPE_CHECKING
import nemos as nmo

if TYPE_CHECKING:
    from .model import BasisNonlinearity


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
    """
    basis = nmo.basis.BSplineEval(
        n_basis_funcs=n_basis_funcs, order=2, bounds=(0.0, 1.0), fill_value=0.0
    )
    return BasisNonlinearity(basis, fill_strategy)