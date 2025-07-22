import numpy as np
from scipy.signal import correlate
from .base_backend import BackendBase
from typing import Literal


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

    def convolve(self, x, kernel, padding: Literal["same", "valid"] = "same"):
        """
        Perform convolution operation.

        Parameters
        ----------
        x :
            Input tensor of shape [batch, 1, time, height, width].
        kernel :
            Kernel tensor of shape [n_kernels, 1, kT, kH, kW].
        padding :
            Padding mode.
        """
        if padding not in ["same", "valid"]:
            raise ValueError(f"Unsupported padding: {padding}")
        x_repeated = x.repeat(1, kernel.shape[0], 1, 1, 1)
        conv = self.lib.nn.functional.conv3d(
            x_repeated, kernel, padding=padding, groups=kernel.shape[0]
        )
        return conv

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

    def convert_array(self, arr: np.ndarray, dtype=None):
        """
        Convert a NumPy array to torch tensor with optional dtype.
        """
        out = self.lib.tensor(arr)
        if dtype is not None:
            out = self.set_dtype(out, dtype)
        return out

    def set_dtype(self, arr, dtype):
        """
        Set the data type of the array.
        """
        if isinstance(dtype, (np.dtype, type)):
            dtype = self.lib.from_numpy(np.array([], dtype=dtype)).dtype
        return arr.to(dtype)

    def minmax_norm(self, x):
        """
        Normalize the input tensor to the range [-1, 1].
        """
        x_min = x.min()
        x_max = x.max()
        return 2 * (x - x_min) / (x_max - x_min) - 1


class JaxBackend(BackendBase):
    def __init__(self):
        import jax.numpy as jnp
        from jax import random, lax, config

        config.update("jax_enable_x64", True)

        self.lib = jnp
        self._random = random
        self._lax = lax
        self.name = "jax"
        self._key = random.PRNGKey(0)

    def randn(self, shape, key: int = None):
        """
        Generate random numbers with a normal distribution.
        """
        if key is None:
            # split to get a new key and update internal state
            self._key, subkey = self._random.split(self._key)
        else:
            subkey = self._random.PRNGKey(key)
        return self._random.normal(subkey, shape)

    def convolve(self, x, kernel, padding: Literal["same", "valid"] = "same"):
        """
        Perform convolution operation.

        Parameters
        ----------
        x :
            Input tensor of shape [batch, 1, time, height, width].
        kernel :
            Kernel tensor of shape [n_kernels, 1, kT, kH, kW].
        padding :
            Padding type.
        """
        if padding not in ["same", "valid"]:
            raise ValueError(f"Unsupported padding: {padding}")
        padding = padding.upper()
        x_repeated = self.lib.repeat(x, kernel.shape[0], axis=1)
        conv = self._lax.conv_general_dilated(
            lhs=x_repeated,
            rhs=kernel,
            window_strides=(1, 1, 1),
            padding=padding,
            dimension_numbers=("NCTHW", "OITHW", "NCTHW"),
            feature_group_count=kernel.shape[0],
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

    def convert_array(self, arr: np.ndarray, dtype=None):
        """
        Convert a NumPy array to jax array with optional dtype.
        """
        out = self.lib.asarray(arr)
        if dtype is not None:
            out = self.set_dtype(out, dtype)
        return out

    def set_dtype(self, arr, dtype):
        """
        Set the data type of the array.
        """
        return arr.astype(dtype)

    def minmax_norm(self, x):
        """
        Normalize the input tensor to the range [-1, 1].
        """
        x_min = x.min()
        x_max = x.max()
        return 2 * (x - x_min) / (x_max - x_min) - 1


class NumpyBackend(BackendBase):
    """
    Backend class for NumPy operations.

    This class provides methods for NumPy array operations.
    """

    def __init__(self):
        self.lib = np
        self.name = "numpy"

    def convolve(self, x, kernel, padding: Literal["same"] = "same"):
        """
        Perform convolution operation.

        VALID padding has not yet been implemented for this backend.

        Parameters
        ----------
        x :
            Input tensor of shape [batch, 1, time, height, width].
        kernel :
            Kernel tensor of shape [n_kernels, 1, kT, kH, kW].
        padding :
            Padding type.
        """
        batch_size, _, t, h, w = x.shape
        n_kernels, _, _, _, _ = kernel.shape
        x_repeated = np.repeat(x, n_kernels, axis=1)
        output = np.zeros((batch_size, n_kernels, t, h, w))

        for b in range(batch_size):
            for k in range(n_kernels):
                input_vol = x_repeated[b, k]
                kernel_vol = kernel[k, 0]
                conv = correlate(input_vol, kernel_vol, mode=padding, method="auto")

                if padding == "same":
                    # Calculate the slices for each dimension based on the expected and actual shapes
                    slices = []
                    original_shape = (t, h, w)
                    for actual, expected in zip(conv.shape, original_shape):
                        if actual > expected:
                            start = (actual - expected) // 2
                            end = start + expected
                            slices.append(slice(start, end))
                        else:
                            slices.append(slice(0, actual))

                    conv = conv[tuple(slices)]

                output[b, k] = conv
        return output

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

    def convert_array(self, arr: np.ndarray, dtype=None):
        """
        Convert a NumPy array to numpy array with optional dtype.
        """
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def set_dtype(self, arr, dtype):
        """
        Set the data type of the array.
        """
        return arr.astype(dtype)

    def minmax_norm(self, x):
        """
        Normalize the input tensor to the range [-1, 1].
        """
        x_min = x.min()
        x_max = x.max()
        return 2 * (x - x_min) / (x_max - x_min) - 1


def get_backend(backend_name: Literal["jax", "torch", "numpy"]) -> BackendBase:
    """
    Get the backend instance based on the specified package.

    Parameters
    ----------
    backend_name :
        The name of the backend to use.

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
