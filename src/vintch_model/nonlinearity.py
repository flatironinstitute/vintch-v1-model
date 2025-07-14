from typing import Any
from einops import einsum
from .backend_config import BackendBase
import numpy as np


class TentNonlinearity:
    """
    Tent-based nonlinearity.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions to use. Must be a positive integer.
    soft :
        Defines whether to use soft or hard tent nonlinearity.
        If True, enables soft nonlinearity.
    backend_class :
        Computational backend to use.
    """

    def __init__(self, backend_class: BackendBase, n_basis_funcs: int = 25):
        if n_basis_funcs < 1:
            raise ValueError("basis functions must be a positive integer.")
        self.n_basis_funcs = n_basis_funcs
        self.backend = backend_class
        self.weights = self.backend.randn(n_basis_funcs)

    @property
    def weights(self):
        """
        Get the weights of the nonlinearity.
        """
        return self._weights

    @weights.setter
    def weights(self, value):
        """
        Set the weights of the nonlinearity.
        """
        self._weights = value

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def compute_basis(self, x):
        """
        Compute tent basis features for input `x`.

        Parameters
        ----------
        x :
            Scalar or array of values.

        Returns
        -------
        Basis activations of shape (n_samples, n_basis_funcs).
        """

        flat_x = x.flatten()
        x_min, x_max = self.backend.lib.min(flat_x), self.backend.lib.max(flat_x)

        # Create a grid of basis centers
        grid = self.backend.lib.linspace(x_min, x_max, self.n_basis_funcs)

        # Compute the tent basis features
        tent_width = grid[1] - grid[0]

        # Calculate the absolute difference and normalize to the tent width
        diff = self.backend.lib.abs(flat_x[:, None] - grid[None, :])
        diff = diff / tent_width

        # Apply the tent function
        values = 1 - diff

        # Ensure non-negative values
        zero_array = self.backend.get_array(np.array([0.0]), dtype=x.dtype)
        features = self.backend.lib.maximum(values, zero_array)
        return features

    def forward(self, x: Any) -> Any:
        """
        Apply the nonlinearity and weights to the input.

        Parameters
        ----------
        x :
            Input data, can be a scalar, NumPy array, JAX tensor, or Torch tensor.

        Returns
        -------
        Output after applying the nonlinearity and weights, reshaped to match input shape.
        """
        original_shape = x.shape

        features = self.compute_basis(x)

        output_flat = einsum(
            features, self.weights, "extended_dim n_basis, n_basis -> extended_dim"
        )
        return output_flat.reshape(original_shape)
