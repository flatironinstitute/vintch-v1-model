import numpy as np


def create_gabor_filter(
    frequency=0.25,
    theta=np.pi / 4,
    sigma_x=3.0,
    sigma_y=3.0,
    n_stds=3,
    offset=0,
    size=None,
):
    """
    Generate a Gabor filter.

    Returns:
        gabor :
            The generated complex Gabor filter.
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

    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    return g


def generate_grating(size=10, spatial_freq=10, orientation=0, phase=0):
    """
    Generate an oriented sinusoidal grating or a stack of gratings if orientation or phase is an array.

    Parameters:
        size (int): Width and height of the square image (in pixels)
        spatial_freq (float): Spatial frequency (cycles per image)
        orientation (float or array-like): Orientation of grating (in degrees, counter-clockwise)
        phase (float or array-like): Phase shift (in degrees)

    Returns:
        2D NumPy array or 3D NumPy array: Single grating or stack of gratings
    """
    # Ensure orientation and phase are arrays for consistent processing
    orientation = np.atleast_1d(orientation)
    phase = np.atleast_1d(phase)

    # Create a coordinate grid centered at (0,0)
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    # Generate gratings for all combinations of orientation and phase
    gratings = []
    for ori in orientation:
        for ph in phase:
            gradient = np.sin(ori * np.pi / 180) * x - np.cos(ori * np.pi / 180) * y
            grating = np.sin((2 * np.pi * gradient) / spatial_freq + (ph * np.pi) / 180)
            grating_norm = (grating - np.min(grating)) / (
                np.max(grating) - np.min(grating)
            )
            gratings.append(grating_norm)

    # Stack gratings along a new axis if multiple gratings are generated
    return np.stack(gratings, axis=0) if len(gratings) > 1 else gratings[0]


def train_validation_split(data, output, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    train_output = output[:split_index]
    val_output = output[split_index:]
    return train_data, val_data, train_output, val_output
