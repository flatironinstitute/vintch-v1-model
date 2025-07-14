import numpy as np
from .base_backend import BackendBase


class TorchBackend(BackendBase):
    """

    Backend class for PyTorch operations.

    This class provides methods for tensor operations, convolution,
    add testsss
    """

    def __init__(self):
        import torch

        self.lib = torch
        self.name = "torch"

    def randn(self, shape, key: int = None):
        """
        Generate random numbers with a normal distribution.
        """
        if key is not None:
            self.lib.manual_seed(key)
        return self.lib.randn(shape)

    def convolve(self, x, kernel, padding="same"):
        """
        Perform convolution operation.
        """
        return self.lib.nn.functional.conv3d(x, kernel, padding=padding)

    def check_input_type(self, x):
        """
        Check the type of the input array.
        """
        assert isinstance(
            x, self.lib.Tensor
        ), f"Expected input to be a torch Tensor, got {type(x)} instead"

    def to_numpy(self, x):
        """
        Convert the input array to a NumPy array.
        """
        return x.detach().cpu().numpy()

    def setitem(self, arr, index, value):
        """
        Set a value in the array at the specified index.
        """
        arr[index] = value
        return arr

    def get_array(self, arr: np.ndarray, dtype=None):
        """
        Convert a NumPy array to the backend-specific array type.
        """
        if dtype is not None:
            return self.lib.tensor(arr, dtype=dtype)
        return self.lib.tensor(arr)


class JaxBackend(BackendBase):
    def __init__(self):
        import jax.numpy as jnp
        from jax import random, lax

        self.lib = jnp
        self.random = random
        self.lax = lax
        self.name = "jax"
        self.key = random.PRNGKey(0)

    def randn(self, shape, key: int = None):
        """
        Generate random numbers with a normal distribution.
        """
        if key is None:
            # split to get a new key and update internal state
            self.key, subkey = self.random.split(self.key)
        else:
            subkey = self.random.PRNGKey(key)
        return self.random.normal(subkey, shape)

    def convolve(self, x, kernel, padding="SAME"):
        """
        Perform convolution operation.
        """
        conv = self.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding=padding,
        )
        return conv

    def check_input_type(self, x):
        """
        Check the type of the input array.
        """
        assert isinstance(
            x, self.lib.ndarray
        ), f"Expected input to be a jax ndarray, got {type(x)} instead"

    def to_numpy(self, x):
        """
        Convert the input array to a NumPy array.
        """
        return np.asarray(x)

    def setitem(self, arr, index, value):
        """
        Set a value in the array at the specified index.
        """
        return arr.at[index].set(value)

    def get_array(self, arr: np.ndarray, dtype=None):
        """
        Convert a NumPy array to the backend-specific array type.
        """
        if dtype is not None:
            return self.lib.asarray(arr, dtype=dtype)
        return self.lib.asarray(arr)


class NumpyBackend(BackendBase):
    """
    Backend class for NumPy operations.

    This class provides methods for NumPy array operations.
    """

    def __init__(self):
        self.lib = np
        self.name = "numpy"

    def convolve(self, x, kernel, padding="same"):  # MISSING IMPLEMENTATION
        """
        Perform convolution operation.
        """
        pass

    def check_input_type(self, x):
        """
        Check the type of the input array.
        """
        assert isinstance(
            x, self.lib.ndarray
        ), f"Expected input to be a numpy ndarray, got {type(x)} instead"

    def to_numpy(self, x):
        """
        Convert the input array to a NumPy array.
        """
        return x

    def randn(self, shape, key: int = None):
        """
        Generate random numbers with a normal distribution.
        """
        if key is not None:
            np.random.seed(key)
        return self.lib.random.randn(*shape)

    def setitem(self, arr, index, value):
        """
        Set a value in the array at the specified index.
        """
        arr[index] = value
        return arr

    def get_array(self, arr: np.ndarray, dtype=None):
        """
        Convert a NumPy array to the backend-specific array type.
        """
        if dtype is not None:
            return arr.astype(dtype)
        return arr


def get_backend(backend_name: str):
    """
    Get the backend instance based on the specified package.

    Parameters
    ----------
    backend_name :
        The name of the backend to use ('jax' or 'torch').

    Returns
    -------
    Backend instance
        An instance of the specified backend class.
    """
    if backend_name == "jax":
        return JaxBackend()
    elif backend_name == "torch":
        return TorchBackend()
    elif backend_name == "numpy":
        return NumpyBackend()
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")
