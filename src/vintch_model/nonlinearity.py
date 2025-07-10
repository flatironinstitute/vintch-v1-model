from typing import Any
from einops import einsum
from .backend_config import get_backend

# type_ArrayLike = Union[np.ndarray, "jax.numpy.ndarray", "torch.Tensor"]


class TentNonlinearity:
    """
    Tent-based nonlinearity.

    Supports jax, or torch backends.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions to use.
    soft :
        Defines whether to use soft or hard tent nonlinearity.
        If True, enables soft nonlinearity. Defaults to False.
    backend :
        Computational backend to use, either 'jax' or 'torch'. Defaults to 'jax'.
    """

    def __init__(
        self, n_basis_funcs: int = 25, soft: bool = False, backend: str = "jax"
    ):
        if n_basis_funcs < 1:
            raise ValueError("basis functions must be at least 1.")
        self.n_basis_funcs = n_basis_funcs
        self.soft = soft
        self.backend = get_backend(backend)
        self.grid = self.backend.lib.linspace(0, 1, n_basis_funcs)
        self.weights = self.backend.randn(n_basis_funcs)
        # self.weights = self.backend.lib.ones(n_basis_funcs)

    def _compute_hard_basis(self, x, grid):
        """
        Compute hard tent basis features for input x and grid.
        """
        features = self.backend.lib.zeros((len(x), len(grid)))
        for i, val in enumerate(x):
            idx = int(self.backend.lib.argmin(self.backend.lib.abs(grid - val)))
            if self.backend.name == "jax":
                features = features.at[i, idx].set(1)
            else:
                features[i, idx] = 1
        return features

    def _compute_soft_basis(self, x, grid):
        """
        Compute soft tent basis features for input x and grid.
        """
        delta = grid[1] - grid[0]
        diff = self.backend.lib.abs(x[:, None] - grid[None, :])
        diff = diff / delta
        val = 1 - diff
        if self.backend.name == "torch":
            features = self.backend.lib.maximum(
                val, self.backend.lib.tensor(0, device=x.device, dtype=x.dtype)
            )
        else:
            features = self.backend.lib.maximum(val, 0)
        return features

    def compute_basis(self, x, grid=None):
        """
        Compute basis features for input `x` using the specified grid.

        Parameters
        ----------
        x :
            Scalar or array of values.
        grid :
            Custom grid for basis functions. If None, uses the default grid.

        Returns
        -------
        Basis activations of shape (n_samples, n_basis_funcs).
        """
        if grid is None:
            grid = self.grid

        x = self.backend.atleast_1d(x)
        if self.soft:
            return self._compute_soft_basis(x, grid)
        else:
            return self._compute_hard_basis(x, grid)

    def evaluate_on_grid(self, n_points: int):
        """
        Evaluate basis features on a uniform grid of n_points between 0 and 1.

        Parameters
        ----------
        n_points :
            Number of points on the grid.

        Returns
        -------
        Tuple of (x_vals, y_vals) where x_vals are the grid points and y_vals are the basis activations.
        """
        x_vals = self.backend.lib.linspace(0, 1, n_points)
        y_vals = self.compute_basis(x_vals)
        return x_vals, y_vals

    def evaluate(self, x: Any):
        """
        Compute basis features for input `x` over a dynamic grid spanning `x`.

        Parameters
        ----------
        x :
            Scalar or array of values (NumPy, JAX, or Torch tensor).

        Returns
        -------
        Basis activations of shape (n_samples, n_basis_funcs).
        """
        x = self.backend.atleast_1d(x)
        x_min, x_max = self.backend.lib.min(x), self.backend.lib.max(x)
        dynamic_grid = self.backend.lib.linspace(x_min, x_max, self.n_basis_funcs)

        if self.soft:
            return self._compute_soft_basis(x, dynamic_grid)
        else:
            return self._compute_hard_basis(x, dynamic_grid)

    def apply_weights(self, features: Any):
        return einsum(features, self.weights, "batch features, features -> batch")

    def transform(self, x: Any) -> Any:
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
        x = self.backend.atleast_1d(x)
        original_shape = x.shape
        flat_x = x.flatten()

        features = self.evaluate(flat_x)
        output_flat = self.apply_weights(features)
        return output_flat.reshape(original_shape)


def tent_nonlinearity_jax_to_torch(jax_instance):
    """
    Convert a TentNonlinearity instance from JAX to Torch backend.
    """
    torch_instance = TentNonlinearity(
        n_basis_funcs=jax_instance.n_basis_funcs,
        soft=jax_instance.soft,
        backend="torch",
    )

    torch_instance.weights = torch_instance.backend.lib.tensor(
        jax_instance.backend.to_numpy(jax_instance.weights)
    )
    torch_instance.grid = torch_instance.backend.lib.tensor(
        jax_instance.backend.to_numpy(jax_instance.grid)
    )
    return torch_instance


def tent_nonlinearity_torch_to_jax(torch_instance):
    """
    Convert a TentNonlinearity instance from Torch to JAX backend.
    """
    jax_instance = TentNonlinearity(
        n_basis_funcs=torch_instance.n_basis_funcs,
        soft=torch_instance.soft,
        backend="jax",
    )
    jax_instance.weights = jax_instance.backend.lib.array(
        torch_instance.backend.to_numpy(torch_instance.weights)
    )
    jax_instance.grid = jax_instance.backend.lib.array(
        torch_instance.backend.to_numpy(torch_instance.grid)
    )
    return jax_instance
