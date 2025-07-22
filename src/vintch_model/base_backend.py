import abc
import numpy as np


class BackendBase(abc.ABC):
    """
    Base class for backend operations.
    """

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def convolve(self, x, kernel, padding):
        pass

    @abc.abstractmethod
    def check_input_type(self, x):
        pass

    @abc.abstractmethod
    def to_numpy(self, x):
        pass

    @abc.abstractmethod
    def randn(self, shape, key: int = None):
        pass

    @abc.abstractmethod
    def setitem(self, arr, index, value):
        pass

    @abc.abstractmethod
    def convert_array(self, arr: np.ndarray, dtype=None):
        pass

    @abc.abstractmethod
    def set_dtype(self, arr, dtype):
        pass

    @abc.abstractmethod
    def minmax_norm(self, x):
        pass
