from typing import TypeVar, Generic, Literal
import einops
from .backend_config import BackendBase
import numpy as np

Tensor = TypeVar("Tensor")


class TentNonlinearity(Generic[Tensor]):
    """
    Tent-based nonlinearity.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions to use. Must be a positive integer.
    backend_instance :
        Backend to use. Must be an instance of BackendBase.
    nonlinearity_type :
        Type of the nonlinearity function, either 'relu' or 'quadratic'.
    """

    def __init__(
        self,
        backend_instance: BackendBase,
        n_basis_funcs: int = 15,
        nonlinearity_type: Literal["relu", "quadratic"] = "relu",
    ):
        if n_basis_funcs < 1:
            raise ValueError("basis functions must be a positive integer.")

        self._n_basis_funcs = n_basis_funcs
        self._backend = backend_instance

        self._build_target_output = (
            self._build_relu_target_output
            if nonlinearity_type == "relu"
            else self._build_quadratic_target_output
        )
        self._initialize_basis_functions()
        self._initialize_weights()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

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

    @property
    def backend(self) -> BackendBase:
        """
        Get the backend instance.
        """
        return self._backend

    @backend.setter
    def backend(self, backend_instance: BackendBase):
        self._backend = backend_instance

    @property
    def grid(self) -> Tensor:
        """
        Get the grid used for the tent basis functions.
        """
        return self._grid

    def _initialize_basis_functions(self):
        """
        Initialize the basis functions for the tent nonlinearity.
        """
        self._grid = self._backend.lib.linspace(-1, 1, self._n_basis_funcs)

    def _build_relu_target_output(self):
        """Build target output for relu mode."""
        return np.maximum(0, self._grid)

    def _build_quadratic_target_output(self):
        """Build target output for quadratic mode."""
        return np.power(self._grid, 2)

    def _initialize_weights(self):
        """
        Initialize the weights of the nonlinearity using least squares.

        Returns
        -------
        Initialized weights.
        """
        initial_features = self._backend.to_numpy(self._tents_transform(self._grid))

        target_output = self._build_target_output()

        weights = np.linalg.lstsq(initial_features, target_output, rcond=None)[0]
        self._weights = self._backend.get_array(weights)

    def _tents_transform(self, x: Tensor) -> Tensor:
        """Compute tent basis features for input x."""
        flat_x = x.flatten()
        tent_width = self._grid[1] - self._grid[0]

        # Normalize x to the range [-1, 1]
        x_normalized = self._backend.minmax_norm(flat_x)

        # Calculate the absolute difference and normalize to the tent width
        diff = self._backend.lib.abs(x_normalized[:, None] - self._grid[None, :])
        diff = diff / tent_width

        # Apply the tent function
        values = 1 - diff

        # Ensure non-negative values
        zero_array = self._backend.get_array(np.array([0.0]), dtype=x.dtype)
        features = self._backend.lib.maximum(values, zero_array)
        return features

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the nonlinearity and weights to the input.

        Parameters
        ----------
        x :
            Input data.

        Returns
        -------
        Output after applying the nonlinearity and weights, reshaped to match input shape.
        """
        self._backend.check_input_type(x)
        self._weights = self._backend.set_dtype(self._weights, x.dtype)

        original_shape = x.shape

        features = self._tents_transform(x)

        output_flat = einops.einsum(
            features, self._weights, "extended_dim n_basis, n_basis -> extended_dim"
        )

        return output_flat.reshape(original_shape)
