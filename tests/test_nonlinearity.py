import pytest
from vintch_model.backend_config import (
    TorchBackend,
    JaxBackend,
    NumpyBackend,
)
from vintch_model.nonlinearity import TentNonlinearity


class TestNonlinearity:
    @pytest.mark.parametrize(
        "backend_cls",
        [
            TorchBackend,
            JaxBackend,
            NumpyBackend,
        ],
    )
    @pytest.mark.parametrize(
        "nonlinearity_mode",
        ["relu", "quadratic"],
    )
    def test_reshapes(self, backend_cls, nonlinearity_mode):
        backend = backend_cls()
        non_linearity = TentNonlinearity(
            backend_instance=backend, nonlinearity_mode=nonlinearity_mode
        )
        x = non_linearity._tent_centers
        x_np = backend.to_numpy(x)

        non_lin_output = non_linearity(x)

        if nonlinearity_mode == "relu":
            expected_output = x_np.clip(min=0)
        elif nonlinearity_mode == "quadratic":
            expected_output = x_np**2

        assert backend.to_numpy(non_lin_output) == pytest.approx(
            expected_output, abs=1e-5
        )
