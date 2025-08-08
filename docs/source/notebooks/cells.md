---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: vintch-v1-model (3.12.10)
    language: python
    name: python3
---

:::

(simple-complex)=
# Simple and Complex cell simulations

In the primary visual cortex (V1), neurons respond selectively to visual stimuli such as oriented edges and gratings. The classic distinction is:

- **Simple cells**, which respond strongly to specific orientations and phases of stimuli. They have distinct ON and OFF subregions in their receptive fields, making their responses highly dependent on stimulus position and phase.
- **Complex cells**, which also respond to specific orientations but are largely invariant to the phase or exact position of the stimulus. Their receptive fields are thought to pool over the outputs of multiple simple cells.

Research has shown that this dichotomy is an oversimplification: V1 neurons exhibit a continuum of behaviors between these two extremes. Some neurons show partial phase sensitivity, falling somewhere between classic simple and complex cell definitions.

In this tutorial, we use the Vintch model to simulate **simple**, **complex**, and **intermediate** cells. We'll examine their responses to visual gratings varying in **phase** and **orientation**.

```python
import numpy as np
from vintch_model.subunit_model import SubunitModel
from vintch_model.utils import create_gabor_filter, create_gaussian_map, generate_grating
from vintch_model.plotting import *
```

**Generate stimulus**

To simulate visual input, we generate sinusoidal grating patterns with varying phase and orientation. These will be used to probe how our model neurons respond to different visual features.

```python
input_size = 15
spatial_freq = 5
orientation = 45

phase_array = np.linspace(0, 360, 17)
orientation_array = np.linspace(0, 360, 17)

phase_grating = generate_grating(size=input_size, spatial_freq=spatial_freq, orientation=orientation, phase=phase_array)
phase_grating = phase_grating.reshape((-1, 1, 1, input_size, input_size))

orientation_grating = generate_grating(size=input_size, spatial_freq=spatial_freq, orientation=orientation_array, phase=0)
orientation_grating = orientation_grating.reshape((-1, 1, 1, input_size, input_size))
```

We create a gabor filter that we will use as a common convolutional filter for the three cell types models. 

```python
kernel_size = 7
pooling_shape = (1, input_size, input_size)
kernel_shape = (1, kernel_size, kernel_size)

gabor = create_gabor_filter(frequency=0.25, theta=-np.pi / 4, sigma_x=2.0, sigma_y=2.0, offset=np.pi, size=(kernel_size, kernel_size)).reshape((1, 1, 1, kernel_size, kernel_size))

plot_kernels(gabor.real, figsize=(2,2))
```

### Simple Cell

Simple cells are phase-selective and orientation-selective. We'll use a narrow Gabor kernel and a small Gaussian pooling map to simulate spatial specificity.

```python
simple_cell = SubunitModel(n_basis_funcs=200, backend="numpy", pooling_shape=pooling_shape,
                         n_channels=1, is_channel_excitatory=[True], subunit_kernel_shape=kernel_shape)

simple_weights = create_gaussian_map(shape=(1, 1, input_size, input_size), sigma=0.5)

simple_cell.kernels = gabor.real
simple_cell.pooling_bias = np.array((0,))
simple_cell.pooling_weights = simple_weights.reshape((1, 1, input_size, input_size))

plot_pooling_weights(simple_cell.pooling_weights, figsize=(2,2))
```

Let's evaluate the model response to a phase-variable and an orientation-variable grating input.

```python
simple_cell_phase_output = simple_cell(phase_grating)
plot_function(phase_array, simple_cell_phase_output, figsize=(5, 2))

simple_cell_orientation_output = simple_cell(orientation_grating)
plot_function(orientation_array, simple_cell_orientation_output, figsize=(5, 2))
```

As expected, the output shows strong tuning to phase, peaking at a specific phase and dropping off at others. To remove constant offsets and focus on tuning shape, we subtract the cell’s baseline response (its response to the first orientation) by adjusting the bias parameter.

```python
simple_cell.pooling_bias = np.array(-simple_cell_orientation_output[0],)

simple_cell_phase_output = simple_cell(phase_grating)
simple_cell_orientation_output = simple_cell(orientation_grating)

plot_function(phase_array, simple_cell_phase_output, figsize=(5, 2))
plot_function(orientation_array, simple_cell_orientation_output, figsize=(5, 2))
```

The resulting tuning curves are consistent with the expected behavior of a simple cell, showing clear selectivity to both phase and orientation.

### Complex Cell

To simulate a complex cell, we use the same Gabor kernel but apply a broader Gaussian pooling map and a squaring nonlinearity, making the cell selective to orientation but largely insensitive to phase.

```python
complex_cell = SubunitModel(n_basis_funcs=200, backend="numpy", pooling_shape=(1, input_size, input_size),
                         n_channels=1, is_channel_excitatory=[False], subunit_kernel_shape=(1, kernel_size, kernel_size))

complex_cell.kernels = gabor.real
complex_cell.pooling_bias = np.array((0,))

complex_weights = create_gaussian_map(shape=(1, 1, input_size, input_size), sigma=10)
complex_cell.pooling_weights = complex_weights.reshape((1, 1, input_size, input_size))

plot_pooling_weights(complex_cell.pooling_weights, figsize=(2,2))
```

And we present the phase and orientation varying stimulus.

```python
complex_cell_phase_output = complex_cell(phase_grating)
plot_function(phase_array, complex_cell_phase_output, figsize=(5, 2))

complex_cell_orientation_output = complex_cell(orientation_grating)
plot_function(orientation_array, complex_cell_orientation_output, figsize=(5, 2))
```

The phase tuning curve is relatively flat, while the orientation tuning remains sharp. We substract the baseline similarily to the simple cell example.

```python
complex_cell.pooling_bias = np.array(-complex_cell_orientation_output[0],)

complex_cell_phase_output = complex_cell(phase_grating)
plot_function(phase_array, complex_cell_phase_output, figsize=(5, 2))

complex_cell_orientation_output = complex_cell(orientation_grating)
plot_function(orientation_array, complex_cell_orientation_output, figsize=(5, 2))
```

The result remains largely unchanged due to the smaller response magnitude of the complex cell. As expected, it stays selective to orientation while showing minimal sensitivity to phase.

### Intermediate Cell


The traditional dichotomy between simple and complex cells is simplification, and individual neurons in V1 exist along a continuum, exhibiting varying degrees of phase sensitivity. Here, we simulate an "intermediate" V1 cell that displays moderate phase selectivity—falling between the extremes of simple and complex cells.

We use a moderate sigma value for the Gaussian pooling map and implement a custom nonlinearity function. This nonlinearity applies a quadratic transformation where positive values are squared, while negative values are scaled down, creating asymmetric response.

```python
intermediate_cell = SubunitModel(n_basis_funcs=200, backend="numpy", pooling_shape=pooling_shape,
                        n_channels=1, is_channel_excitatory=[True], subunit_kernel_shape=kernel_shape)

intermediate_weights = create_gaussian_map(shape=(1, 1, input_size, input_size), sigma=1.5)

intermediate_cell.kernels = gabor.real
intermediate_cell.pooling_bias = np.array((0,))
intermediate_cell.pooling_weights = intermediate_weights.reshape((1, 1, input_size, input_size))

plot_pooling_weights(intermediate_cell.pooling_weights, figsize=(2,2))
```

```python
intermediate_cell_phase_output = intermediate_cell(phase_grating)
intermediate_cell_orientation_output = intermediate_cell(orientation_grating)
```

```python
plot_function(phase_array, intermediate_cell_phase_output, figsize=(5, 2))
plot_function(orientation_array, intermediate_cell_orientation_output, figsize=(5, 2))
```

```python
intermediate_cell.pooling_bias = np.array(-intermediate_cell_orientation_output[0],)
```

```python
intermediate_cell_phase_output = intermediate_cell(phase_grating)
plot_function(phase_array, intermediate_cell_phase_output, figsize=(5, 2))

intermediate_cell_orientation_output = intermediate_cell(orientation_grating)
plot_function(orientation_array, intermediate_cell_orientation_output, figsize=(5, 2))
```

The resulting tuning curves for the intermediate cell show partial phase selectivity and strong orientation selectivity, as expected. The response is less phase-dependent than the simple cell but more so than the complex cell model.