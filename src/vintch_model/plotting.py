from matplotlib import pyplot as plt
import os


def plot_kernels(
    kernels, timestep=0, fontsize=12, tick_fontsize=10, figsize=(4, 4), filename=None
):
    """Plot the kernels."""
    n_channels = kernels.shape[0]
    fig, ax = plt.subplots(
        1, n_channels, figsize=(figsize[0] * n_channels, figsize[1]), dpi=150
    )
    for i in range(n_channels):
        if n_channels == 1:
            ax.imshow(kernels[i, 0, timestep, :, :], cmap="gray")
            ax.axis("off")  # Hide ticks for single channel
        else:
            ax[i].imshow(kernels[i, 0, timestep, :, :], cmap="gray")
            ax[i].set_title(f"Kernel {i+1} at time step {timestep}", fontsize=fontsize)
            ax[i].axis("off")  # Hide ticks for each channel
    # Add colorbar with height 0.8 of the axes
    cbar = fig.colorbar(ax.images[0], ax=ax, orientation="vertical", fraction=0.1)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    if filename:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()


def plot_pooling_weights(
    pooling_weights,
    fontsize=12,
    tick_fontsize=10,
    figsize=(4, 4),
    title=None,
    filename=None,
):
    """Plot pooling weights."""
    fig = plt.figure(figsize=figsize, dpi=150)
    ax1 = fig.add_subplot(1, 1, 1)
    im = ax1.imshow(pooling_weights[0, 0], cmap="gray", vmin=0, vmax=1)
    ax1.axis("off")
    if title is not None:
        ax1.set_title(title, fontsize=fontsize)
    cbar = fig.colorbar(im, ax=ax1, orientation="vertical", fraction=0.1)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    if filename:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()


def plot_tuning_curve(
    x,
    response,
    xlabel="Stimulus Feature",
    ylabel="Response",
    title=None,
    vrange=None,
    color="#007acc",
    fontsize=12,
    tick_fontsize=10,
    figsize=(4, 4),
    xticks=None,
    filename=None,
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
        tick_fontsize (int): Font size for axis and colorbar ticks
        figsize (tuple): Figure size (width, height)
        xticks (array-like): Custom x-axis tick positions
        filename (str): Filename to save the plot as SVG in figures directory
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
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
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        max_ticks = fig.get_size_inches()[0]
        if len(x) > max_ticks:
            step = max(1, len(x) // int(max_ticks))
            ax.set_xticks(x[::step])
        else:
            ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    if vrange is not None:
        ax.set_ylim(vrange)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if filename:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()


def plot_grating(
    grating, title=None, fontsize=12, tick_fontsize=10, figsize=(4, 4), filename=None
):
    """
    Plots a grating stimulus or a stack of gratings.

    Parameters:
        grating (numpy.ndarray): The grating stimulus to plot. Can be 2D or 3D (stack of gratings).
        title (str): Title of the plot
        fontsize (int): Font size for labels and title
        tick_fontsize (int): Font size for axis and colorbar ticks
        figsize (tuple): Figure size (width, height)
        filename (str): Filename to save the plot as SVG in figures directory
    """
    if grating.ndim == 2:
        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_subplot(111)
        im = ax.imshow(grating, cmap="gray")
        ax.axis("off")
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        cbar = fig.colorbar(im, ax=ax, fraction=0.1)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        plt.tight_layout()
        if filename:
            save_dir = "figures"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            plt.rcParams["svg.fonttype"] = "none"  # keep text as text, not paths
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.show()
    elif grating.ndim == 3:
        n_gratings = grating.shape[0]
        fig, axes = plt.subplots(
            1, n_gratings, figsize=(figsize[0] * n_gratings, figsize[1]), dpi=150
        )
        if n_gratings == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(grating[i], cmap="gray")
            ax.axis("off")
            if title is not None:
                ax.set_title(f"{title} {i+1}", fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
        cbar = fig.colorbar(im, ax=ax, fraction=0.1)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        plt.tight_layout()
        if filename:
            save_dir = "figures"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.show()


def plot_function(
    x,
    y,
    xlabel="x",
    ylabel="y",
    title=None,
    color="#007acc",
    fontsize=16,
    tick_fontsize=12,
    figsize=(4, 4),
    xticks=None,
    yticks=None,
    filename=None,
    yrange=(-1, 1),
):
    """
    Plots a mathematical function y = f(x) with scientific poster style.

    Parameters:
        x (array-like): x values
        y (array-like): y values
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        title (str): Title of the plot
        color (str): Line and marker color
        fontsize (int): Font size for labels and title
        tick_fontsize (int): Font size for axis ticks
        figsize (tuple): Figure size (width, height)
        xticks (array-like): Custom x-axis tick positions
        yticks (array-like): Custom y-axis tick positions
        filename (str): Filename to save the plot as SVG in figures directory
        yrange (tuple): y-axis limits (min, max), default (-1, 1)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(x, y, linestyle="-", linewidth=3, color=color)
    ax.axvline(
        0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7
    )  # Thin vertical line at x=0
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
    else:
        ax.set_xticks([])
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
    else:
        ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(yrange)
    plt.tight_layout()
    if filename:
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()
