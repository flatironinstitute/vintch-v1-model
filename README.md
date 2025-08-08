# vintch-v1-model
Implementation of V1 model from Vintch et al., 2015

---

**Note:** This implementation is still in progress. Here is the current state of the project, a description of the main branches, and the next steps planned for development.

### Repository Structure & Branches

- **main**: The stable base branch.

- **refine_model**: Contains small improvements to the model, such as switching errors to warnings. This branch also introduces utility and plotting functions like `create_gabor_filter` and `create_gaussian_map`, which are useful for simulating different cell types and visualizing model components.

- **model_fit**: This branch contains the fitting methods implemented within the `subunit_model` script. These methods are designed to fit the model parameters `kernels` and `pooling_weights` to data. I also left a fitting script in `docs/source/examples`, which you can run directly. This script does parameter recovery.

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

For any questions, please contact me at **pacresda@gmail.com**.
