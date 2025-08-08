# vintch-v1-model  
Implementation of the V1 model from Vintch et al., 2015

---

Note: This implementation is a work in progress. Below is the current status of the project.

### Repository Structure

- `utils.py` – Contains helper functions such as `create_gabor_filter` and `create_gaussian_map`, useful for simulating different cell types.
- `plotting.py` – Includes functions for visualizing the components of the model.
- `docs/source/notebooks/` – Includes two notebooks demonstrating how to simulate different cell types and how to apply MAD competition for simple and complex cell models.
- `subunit_model.py` – Implements the model fitting methods. These are designed to optimize the parameters `kernels` and `pooling_weights`.  
  An example fitting script is available in `docs/source/examples/`.

---

## Next Steps

1. **Ensure the fitting procedure can recover all model parameters.**
    - Consider what constraints are needed so that training converges to the desired solution. For example, parametrizing the pooling layer as a (possibly centered) Gaussian distribution.
    - Revisit the original MATLAB code to understand better the details in their fitting procedure.
    - Implement initialization using STC and STA, as we observe that initialization has a significant impact on the solution found.
2. Consider removing the `fit_together` function, since both the paper and our experiments suggest that fitting both parameters together is not beneficial.
3. Update the implementation to allow for spike counts as the observable variable. Currently, the model assumes it observes rates.
4. Try fitting the model to real data, including both orientation and white noise input for simple and complex cells.
5. So far, `utils.py` functions only support numpy objects, they should be adapted to the different backends.

---

For questions or issues, please open an issue on the repository.
