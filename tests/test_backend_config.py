import pytest
import numpy as np
from vintch_model.backend_config import (
    TorchBackend,
    JaxBackend,
    NumpyBackend,
)


@pytest.mark.parametrize(
    "backend_cls",
    [
        TorchBackend,
        JaxBackend,
        NumpyBackend,
    ],
)
class TestBackend:
    @pytest.mark.parametrize(
        "input_shape",
        [
            (2, 3),
            (1, 1, 10, 10, 10),
        ],
    )
    def test_check_input_type(self, backend_cls, input_shape):
        backend = backend_cls()

        x = backend.randn(input_shape)
        backend.check_input_type(x)

    def test_setitem(self, backend_cls):
        backend = backend_cls()

        arr = backend.lib.zeros(3)
        arr2 = backend.setitem(arr, 1, 5.0)
        assert arr2[1] == 5.0

    def test_to_numpy(self, backend_cls):
        backend = backend_cls()

        arr = backend.lib.zeros(3)
        np_arr = backend.to_numpy(arr)
        assert isinstance(np_arr, np.ndarray)

    @pytest.mark.parametrize(
        "shape",
        [(2, 2)],
    )
    def test_randn(self, backend_cls, shape):
        backend = backend_cls()
        rand = backend.randn(shape, key=123)
        assert rand.shape == shape, "Output shape mismatch for randn"

    def test_convolve(self, backend_cls):
        if backend_cls == NumpyBackend:
            pytest.skip("NumpyBackend does not support convolution")
        backend = backend_cls()

        x = backend.lib.ones((1, 1, 5, 5, 5))
        kernel = backend.lib.ones((1, 1, 3, 3, 3))
        out = backend.convolve(x, kernel, padding="same")
        assert out.shape == x.shape, "Output shape mismatch for convolve"


def test_convolve_consistency_across_backends():
    backends = [TorchBackend(), JaxBackend()]
    x_shapes = (1, 1, 25, 25, 25)
    kernel_shapes = (2, 1, 3, 3, 3)

    kernel = np.random.rand(*kernel_shapes)
    x = np.random.rand(*x_shapes)

    outputs = []
    for backend in backends:
        x = backend.get_array(x)
        kernel = backend.get_array(kernel)
        out = backend.convolve(x, kernel, padding="same")
        outputs.append(backend.to_numpy(out))

    assert np.allclose(
        outputs[0], outputs[1]
    ), "Convolution outputs differ across backends"
