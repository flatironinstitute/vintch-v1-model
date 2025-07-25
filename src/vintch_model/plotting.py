from matplotlib import pyplot as plt


def plot_kernels(kernels, timestep=0, fontsize=12, figsize=(3, 3), save_as_svg=None):
    """Plot the kernels."""
    n_channels = kernels.shape[0]
    fig, ax = plt.subplots(
        1, n_channels, figsize=(figsize[0] * n_channels, figsize[1]), dpi=150
    )
    for i in range(n_channels):
        if n_channels == 1:
            ax.imshow(kernels[i, 0, timestep, :, :], cmap="gray")
        else:
            ax[i].imshow(kernels[i, 0, timestep, :, :], cmap="gray")
            ax[i].set_title(f"Kernel {i+1} at time step {timestep}", fontsize=fontsize)
            ax[i].axis("off")

    # Add colorbar
    fig.colorbar(ax.images[0], ax=ax, orientation="vertical", fraction=0.1, pad=0.04)
    plt.tight_layout()
    if save_as_svg:
        plt.savefig(save_as_svg, format="svg")
    plt.show()


def plot_pooling_weights(
    pooling_weights, fontsize=12, figsize=(4, 4), save_as_svg=None
):
    """Plot pooling weights."""
    fig = plt.figure(figsize=figsize, dpi=150)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("Pooling Weights Channel 1", fontsize=fontsize)
    ax1.imshow(pooling_weights[0, 0], cmap="gray", vmin=0, vmax=1)
    fig.colorbar(ax1.images[0], ax=ax1, orientation="vertical", fraction=0.05)
    plt.tight_layout()
    if save_as_svg:
        plt.savefig(save_as_svg, format="svg")
    plt.show()


def plot_tuning_curve(
    x,
    response,
    xlabel="Stimulus Feature",
    ylabel="Response",
    title="Tuning Curve",
    vrange=None,
    color="#007acc",
    fontsize=12,
    figsize=(5, 5),
    xticks=None,
    save_as_svg=None,
):
    """
    Plots a tuning curve for a stimulus feature vs. model/neural response.

    Parameters:
        x (array-like): Stimulus feature values (e.g., orientation, phase)
        response (array-like): Response values for each stimulus
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        title  (str): Title of the plot
        vrange (tuple): y-axis limits (min, max)
        color (str): Line and marker color (default: blue)
        fontsize (int): Font size for labels and title
        figsize (tuple): Figure size (width, height)
        save_as_svg (str): Path to save the plot as an SVG file
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Plot with publication-friendly styling
    ax.plot(
        x,
        response,
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=7,
        color=color,
        markerfacecolor="white",
        markeredgewidth=2,
    )

    # Axis labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1, weight="bold")

    # Add a light horizontal line at y=0
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        # Dynamically adjust x-axis ticks based on figure size
        max_ticks = fig.get_size_inches()[0]  # Allow ~1 tick per inch of figure width
        if len(x) > max_ticks:
            step = max(1, len(x) // int(max_ticks))
            ax.set_xticks(x[::step])
        else:
            ax.set_xticks(x)

    # Tick formatting
    ax.tick_params(axis="x", labelsize=fontsize - 2)
    ax.tick_params(axis="y", labelsize=fontsize - 2)

    # Optional y-axis range
    if vrange is not None:
        ax.set_ylim(vrange)

    # Clean layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_as_svg:
        plt.savefig(save_as_svg, format="svg")
    plt.show()


def plot_grating(
    grating, title="Grating Stimulus", fontsize=12, figsize=(4, 4), save_as_svg=None
):
    """
    Plots a grating stimulus or a stack of gratings.

    Parameters:
        grating (numpy.ndarray): The grating stimulus to plot. Can be 2D or 3D (stack of gratings).
        title (str): Title of the plot
        fontsize (int): Font size for labels and title
        figsize (tuple): Figure size (width, height)
        save_as_svg (str): Path to save the plot as an SVG file
    """
    if grating.ndim == 2:
        # Single grating
        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_subplot(111)
        ax.imshow(grating, cmap="gray")
        ax.axis("off")
        ax.set_title(title, fontsize=fontsize)
        fig.colorbar(ax.imshow(grating, cmap="gray"), ax=ax)
        plt.tight_layout()
        if save_as_svg:
            plt.savefig(save_as_svg, format="svg")
        plt.show()
    elif grating.ndim == 3:
        # Stack of gratings
        n_gratings = grating.shape[0]
        fig, axes = plt.subplots(
            1, n_gratings, figsize=(figsize[0] * n_gratings, figsize[1]), dpi=150
        )
        if n_gratings == 1:
            axes = [axes]  # Ensure axes is iterable for a single grating
        for i, ax in enumerate(axes):
            ax.imshow(grating[i], cmap="gray")
            ax.axis("off")
            ax.set_title(f"{title} {i+1}", fontsize=fontsize)
            fig.colorbar(ax.imshow(grating[i], cmap="gray"), ax=ax)
        plt.tight_layout()
        if save_as_svg:
            plt.savefig(save_as_svg, format="svg")
        plt.show()
