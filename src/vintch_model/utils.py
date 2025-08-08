import numpy as np
from typing import Literal
from .backend_config import get_backend


def create_gabor_filter(
    frequency: float = 0.25,
    theta: float = np.pi / 4,
    sigma_x: float = 3.0,
    sigma_y: float = 3.0,
    n_stds: int = 3,
    offset: float = 0.0,
    size=None,
    backend: Literal["torch", "jax", "numpy"] = "numpy",
):
    """
    Generate a Gabor filter.

    This function mimics the behavior of the `skimage.filters.gabor` function, generating a complex Gabor
    filter, but allowing for custom size and normalized to [0,1] range.

    Parameters
    ----------
    frequency :
        Frequency of the sinusoidal factor.
    theta :
        Orientation of the Gabor function.
    sigma_x :
        Standard deviation of the Gaussian envelope along the x-axis.
    sigma_y :
        Standard deviation of the Gaussian envelope along the y-axis.
    n_stds :
        The linear size of the kernel is n_stds standard deviations
    offset :
        Phase offset of the sinusoidal factor.
    size :
        Size of the filter (height, width).
    backend :
        Computational backend to use: 'torch', 'jax', or 'numpy'.

    Returns
    -------
    gabor :
        The generated complex Gabor filter in the specified backend in the range [-1, 1].
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    if size is None:
        x0 = np.round(max(abs(n_stds * sigma_x * ct), abs(n_stds * sigma_y * st), 1))
        y0 = np.round(max(abs(n_stds * sigma_y * ct), abs(n_stds * sigma_x * st), 1))
    else:
        y0 = (size[0] - 1) // 2
        x0 = (size[1] - 1) // 2

    y, x = np.meshgrid(
        np.arange(-y0, y0 + 1), np.arange(-x0, x0 + 1), indexing="ij", sparse=True
    )
    rotx = x * ct + y * st
    roty = -x * st + y * ct

    g = np.empty(roty.shape, dtype=np.complex128)
    np.exp(
        -0.5 * (rotx**2 / sigma_x**2 + roty**2 / sigma_y**2)
        + 1j * (2 * np.pi * frequency * rotx + offset),
        out=g,
    )
    g *= 1 / (2 * np.pi * sigma_x * sigma_y)

    g = 2 * (g - np.min(g)) / (np.max(g) - np.min(g)) - 1

    backend_instance = get_backend(backend)
    g = backend_instance.convert_array(g)
    return g


def generate_grating(
    size: int = 10,
    spatial_freq: float = 10,
    orientation: float = 0,
    phase: float = 0,
    backend: Literal["torch", "jax", "numpy"] = "numpy",
):
    """
    Generate an oriented sinusoidal grating or a stack of gratings if orientation or phase is an array.

    Parameters
    ----------
    size :
        Size of the grating (height and width).
    spatial_freq :
        Spatial frequency of the grating.
    orientation :
        Orientation(s) of the grating in degrees. Can be a scalar or array.
    phase :
        Phase(s) of the grating in degrees. Can be a scalar or array.
    backend :
        Computational backend to use: 'torch', 'jax', or 'numpy'.

    Returns
    -------
    grating :
        The generated grating or stack of gratings, normalized to [0, 1].
    """
    orientation = np.atleast_1d(orientation)
    phase = np.atleast_1d(phase)

    x, y = np.meshgrid(np.arange(size), np.arange(size))

    gratings = []
    for ori in orientation:
        for ph in phase:
            gradient = np.sin(ori * np.pi / 180) * x - np.cos(ori * np.pi / 180) * y
            grating = np.sin((2 * np.pi * gradient) / spatial_freq + (ph * np.pi) / 180)
            grating_norm = (grating - np.min(grating)) / (
                np.max(grating) - np.min(grating)
            )
            gratings.append(grating_norm)

    result = np.stack(gratings, axis=0) if len(gratings) > 1 else gratings[0]
    backend_instance = get_backend(backend)
    result = backend_instance.convert_array(result)
    return result


def create_gaussian_map(shape: tuple, sigma: float) -> np.ndarray:
    """
    Create a Gaussian map with the given shape and standard deviation.

    Parameters
    ----------
    shape :
        Shape of the Gaussian map (3D or 4D). For 4D, the shape should be (input_channels, time, height, width).
        For 3D, the shape should be (input_channels, height, width).
    sigma :
        Standard deviation of the Gaussian.

    Returns
    -------
    gaussian :
        The generated Gaussian map.
    """
    map_size = shape[-1]
    center = map_size // 2
    if len(shape) == 4:
        x, y, t = np.meshgrid(
            np.arange(shape[3]), np.arange(shape[2]), np.arange(shape[1]), indexing="ij"
        )
        gaussian = np.exp(
            -((x - center) ** 2 + (y - center) ** 2 + (t - center) ** 2)
            / (2 * sigma**2)
        )
    elif len(shape) == 3:
        x, y = np.meshgrid(np.arange(map_size), np.arange(map_size), indexing="ij")
        gaussian = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    else:
        raise ValueError("parameter shape must be 3D or 4D.")

    gaussian /= gaussian.max()

    gaussian = np.expand_dims(gaussian, axis=0)
    return gaussian


def train_validation_split(data, output, split_ratio=0.8):
    """
    Split data and output into training and validation sets.

    Parameters
    ----------
    data :
        Input data to split.
    output :
        Output data to split.
    split_ratio :
        Fraction of data to use for training.

    Returns
    -------
    x_train :
        Training input.
    x_val :
        Validation input.
    y_train :
        Training target.
    y_val :
        Validation target.
    """
    split_index = int(len(data) * split_ratio)
    x_train = data[:split_index]
    x_val = data[split_index:]
    y_train = output[:split_index]
    y_val = output[split_index:]
    return x_train, x_val, y_train, y_val
