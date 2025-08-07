from matplotlib import pyplot as plt
import os

figures_dir = "figures"


def plot_kernels(kernels, timestep=0, figsize=(4, 4), filename=None):
    """
    Plot the kernels of the subunit model.

    Parameters
    ----------
    kernels :
        Kernel tensor with shape [n_channels, 1, time, height, width].
    timestep :
        Time index to visualize.
    figsize :
        Figure size (width, height) for each kernel.
    filename :
        If provided, saves the figure as an SVG in the 'figures' directory with this filename.
    """
    n_channels = kernels.shape[0]
    fig, axes = plt.subplots(
        1, n_channels, figsize=(figsize[0] * n_channels, figsize[1]), dpi=150
    )
    axes = [axes] if n_channels == 1 else axes
    for i, ax in enumerate(axes):
        ax.imshow(kernels[i, 0, timestep], cmap="gray")
        ax.axis("off")
        if n_channels > 1:
            ax.set_title(f"Kernel {i+1} at time step {timestep}", fontsize=10)
    fig.colorbar(
        axes[0].images[0], ax=axes, orientation="vertical", fraction=0.1
    ).ax.tick_params(labelsize=10)
    plt.tight_layout()
    if filename:
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(
            os.path.join(figures_dir, filename), format="svg", bbox_inches="tight"
        )
    plt.show()


def plot_pooling_weights(pooling_weights, figsize=(4, 4), filename=None):
    """
    Plot the pooling weights of the subunit model.

    Parameters
    ----------
    pooling_weights :
        Pooling weights tensor with shape [n_channels, time, height, width] or [n_channels, height, width].
    figsize :
        Figure size (width, height) for the plot.
    filename :
        If provided, saves the figure as an SVG in the 'figures' directory with this filename.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(pooling_weights[0, 0], cmap="gray")
    ax.axis("off")
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.1).ax.tick_params(
        labelsize=10
    )
    plt.tight_layout()
    if filename:
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(
            os.path.join(figures_dir, filename), format="svg", bbox_inches="tight"
        )
    plt.show()


def plot_function(x, response, xlabel="", ylabel="", figsize=(4, 4), filename=None):
    """
    Plot a function response = f(x).

    Parameters
    ----------
    x :
        The x values.
    response :
        The response values corresponding to x.
    xlabel :
        Label for the x-axis.
    ylabel :
        Label for the y-axis.
    figsize :
        Figure size (width, height) for the plot.
    filename :
        If provided, saves the figure as an SVG in the 'figures' directory with this filename.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(
        x,
        response,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=2,
        markeredgewidth=2,
    )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if filename:
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(
            os.path.join(figures_dir, filename), format="svg", bbox_inches="tight"
        )
    plt.show()


def plot_grating(grating, figsize=(4, 4), filename=None):
    """
    Plot a grating stimulus or a stack of gratings.

    Parameters
    ----------
    grating :
        The grating stimulus to plot. Can be 2D (height, width) or 3D (n_gratings, height, width).
    figsize :
        Figure size (width, height) for the plot.
    filename :
        If provided, saves the figure as an SVG in the 'figures' directory with this filename.
    """
    if grating.ndim == 2:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        im = ax.imshow(grating, cmap="gray")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.1).ax.tick_params(labelsize=10)
        plt.tight_layout()
    else:
        n_gratings = grating.shape[0]
        fig, axes = plt.subplots(
            1, n_gratings, figsize=(figsize[0] * n_gratings, figsize[1]), dpi=150
        )
        axes = [axes] if n_gratings == 1 else axes
        for i, ax in enumerate(axes):
            im = ax.imshow(grating[i], cmap="gray")
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.1).ax.tick_params(labelsize=10)
        plt.tight_layout()
    if filename:
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(
            os.path.join(figures_dir, filename), format="svg", bbox_inches="tight"
        )
    plt.show()
