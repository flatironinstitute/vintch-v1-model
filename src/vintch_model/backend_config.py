import numpy as np


class TorchBackend:
    def __init__(self):
        import torch

        self.lib = torch
        self.randn = torch.randn
        self.conv3d = torch.nn.functional.conv3d
        self.name = "torch"

    def atleast_1d(self, x):
        return x if x.dim() > 0 else x.unsqueeze(0)

    def convolve(self, x, kernel):
        return self.lib.nn.functional.conv3d(x, kernel, padding="same")

    def check_input_type(self, x):
        assert isinstance(
            x, self.lib.Tensor
        ), f"Expected input to be a torch Tensor, got {type(x)} instead"

    def to_numpy(self, x):
        return x.detach().cpu().numpy()


class JaxBackend:
    def __init__(self):
        import jax.numpy as jnp
        from jax import random, lax

        self.lib = jnp
        self.random = random
        self.atleast_1d = jnp.atleast_1d
        self.lax = lax
        self.name = "jax"

    def randn(self, n):
        key = self.random.PRNGKey(0)
        return self.random.normal(key, n)

    def convolve(self, x, kernel):
        conv = self.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding="SAME",
        )
        return conv

    def check_input_type(self, x):
        assert isinstance(
            x, self.lib.ndarray
        ), f"Expected input to be a jax ndarray, got {type(x)} instead"

    def to_numpy(self, x):
        return np.array(x)


def get_backend(package: str):
    """
    Get the backend instance based on the specified package.

    Parameters
    ----------
    package :
        The name of the package to use as backend ('jax' or 'torch').

    Returns
    -------
    Backend instance
        An instance of JaxBackend or TorchBackend.
    """
    if package == "jax":
        return JaxBackend()
    elif package == "torch":
        return TorchBackend()
    else:
        raise ValueError(f"Unsupported backend: {package}")
