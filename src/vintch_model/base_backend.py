import numpy as np


class BackendBase:
    """
    Base class for backend operations.
    """

    def __init__(self, name):
        self.name = name

    def convolve(self, x, kernel, padding):
        pass

    def check_input_type(self, x):
        pass

    def to_numpy(self, x):
        pass

    def randn(self, shape, key: int = None):
        pass

    def setitem(self, arr, index, value):
        pass

    def get_array(self, arr: np.ndarray, dtype=None):
        """
        Convert a NumPy array to the backend-specific array type.
        """
        pass
