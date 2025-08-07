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

```python
import numpy as np
import torch
import plenoptic as po
from vintch_model.subunit_model import SubunitModel
from vintch_model.utils import create_gabor_filter, create_gaussian_map, generate_grating
from vintch_model.plotting import *
```

# MAD Competition for a simple and a complex cell models

In this tutorial, we investigate the phase sensitivity of simple and complex cells using the [MAD Competition](https://www.cns.nyu.edu/pub/lcv/wang08-preprint.pdf) method. MAD Competition is a framework for comparing two vision models by iteratively modifying an input stimulus to maximize (or minimize) the difference in response for one model, while constraining the response of the other model to remain unchanged. For a detailed explanation of the method and its implementation, see the [Plenoptic function documentation](https://docs.plenoptic.org/docs/branch/main/tutorials/intro/MAD_Competition_2.html).

In this example, we expect MAD Competition to produce a phase-shifted image: the simple cell model will show a strong change in response, while the complex cell model will remain largely invariant.

We begin by defining the simple and complex cell models. For further details on model construction, refer to the [](simple-complex) notebook.

```python
kernel_size = 7
input_size = 15
orientation = 45

pooling_shape = (1, input_size, input_size)
kernel_shape = (1, kernel_size, kernel_size)

gabor = create_gabor_filter(frequency=0.25, theta=-np.pi / 4, sigma_x=2.0, sigma_y=2.0, offset=np.pi, size=(kernel_size, kernel_size)).reshape((1, 1, 1, kernel_size, kernel_size))
gabor = torch.tensor(gabor.real, dtype=torch.float32)

simple_cell = SubunitModel(n_basis_funcs=200, backend="torch", pooling_shape=(1, input_size, input_size),
                         n_channels=1, is_channel_excitatory=[True], subunit_kernel_shape=(1, kernel_size, kernel_size))

simple_cell_weights = create_gaussian_map(shape=(1, 1, input_size, input_size), sigma=0.25)

simple_cell.kernels = gabor
simple_cell.pooling_bias = torch.tensor((-0.,))
simple_cell.pooling_weights = torch.tensor(simple_cell_weights.reshape((1, 1, input_size, input_size)), dtype=torch.float32)

complex_cell = SubunitModel(n_basis_funcs=200, backend="torch", pooling_shape=(1, input_size, input_size),
                         n_channels=1, is_channel_excitatory=[False], subunit_kernel_shape=(1, kernel_size, kernel_size))

complex_cell_weights = create_gaussian_map(shape=(1, 1, input_size, input_size), sigma=10)

complex_cell.kernels = gabor
complex_cell.pooling_bias = torch.tensor((-0.,))
complex_cell.pooling_weights = torch.tensor(complex_cell_weights.reshape((1, 1, input_size, input_size)), dtype=torch.float32)
```

We define a grating stimulus at the cell's preferred orientation and at the phase that minimizes the simple cell's response. This grating serves as our initial input, allowing us to probe the phase sensitivity of both models. The simple cell is expected to be highly sensitive to phase, while the complex cell should be relatively invariant.

```python
x = generate_grating(size=input_size, spatial_freq=5, orientation=45, phase=90)
x = x.reshape((1, 1, input_size, input_size))
x = torch.tensor(x, dtype=torch.float32)

plot_grating(x[0], figsize=(2,2))

simple_cell_output = simple_cell(x)
complex_cell_output = complex_cell(x)

print("Simple Cell Output:", simple_cell_output)
print("Complex Cell Output:", complex_cell_output)
```

Next, we apply the MAD Competition algorithm.

```python
simple_cell_metric = lambda x, y: po.metric.model_metric(x,y, simple_cell)
complex_cell_metric = lambda x, y: po.metric.model_metric(x,y, complex_cell)

mad = po.synth.MADCompetition(x, optimized_metric=simple_cell_metric, reference_metric=complex_cell_metric, minmax="max", metric_tradeoff_lambda=100, allowed_range=(0, 1))
mad.setup(optimizer_kwargs={"lr":.01})
mad.synthesize(max_iter=300, store_progress=True)

fig = po.synth.mad_competition.plot_synthesis_status(mad)
```

The resulting image from MAD Competition reveals the distinct properties of the two models. The center of the image becomes an anti-phase version of the original grating, demonstrating the phase sensitivity of the simple cell. In contrast, the perophery is a blurred version of the original stimulus, which can be explained by the centered and very localized pooling map of the simple cell.