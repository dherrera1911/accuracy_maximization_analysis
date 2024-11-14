import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

from ..data_wrangle import statistics_dim_subset
from .colors import get_class_rgba, get_normalized_color_map


def single_ellipse(center, covariance, ax, color="black"):
    """
    Plot an ellipse with a given center and covariance matrix.

    Parameters
    ----------
    center : torch.Tensor
        Center of the ellipse. Shape (2).
    covariance : torch.Tensor
        Covariance matrix of the ellipse. Shape (2, 2).
    ax : matplotlib.axes.Axes
        Axes to plot the ellipse.
    color : str, optional
        Color of the ellipse. The default is 'black'.
    """
    eig_val, eig_vec = torch.linalg.eigh(covariance)
    # Get the angle of the ellipse main axis
    angle = torch.atan2(eig_vec[1, 0], eig_vec[0, 0])
    # Get the length of the axes
    scale = torch.sqrt(eig_val)
    # Plot the ellipse
    ellipse = patches.Ellipse(
        xy=center,
        width=scale[0] * 4,
        height=scale[1] * 4,
        angle=angle * 180 / np.pi,
        color=color,
    )
    ellipse.set_facecolor("none")
    ellipse.set_linewidth(3)
    ax.add_patch(ellipse)


def statistics_ellipses(
    means, covariances, filter_pair=(0, 1), ax=None, values=None,
    classes_plot=None, color_map="viridis", legend_type='none', **kwargs
):
    """
    Plot the ellipses of the filter response statistics across classes.

    Parameters
    ----------
    means : torch.Tensor
        Means of the filter responses. Shape (n_classes, n_filters).
    covariances : torch.Tensor
        Covariances of the filter responses.
        Shape (n_classes, n_filters, n_filters).
    filter_pair : tuple of int, optional
        Pair of filters to plot. The default is [0, 1].
    ax : matplotlib.axes.Axes, optional
        Axes to plot the ellipses. If None, a new figure is created.
        The default is None.
    values : torch.Tensor, optional
        Values to color code the ellipses. Each value corresponds to a
        class. The default is linearly spaced values between -1 and 1.
    classes_plot : list, optional
        List of classes to plot. The default is all classes.
    color_map : str or matplotlib.colors.Colormap, optional
        Color map to use for the ellipses. The default is 'viridis'.
    legend_type : str, optional
        Type of legend to add: 'none', 'continuous', 'discrete'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the scatter plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    if classes_plot is None:
        classes_plot = np.arange(means.shape[0])

    if values is None:
        values = np.arange(len(classes_plot))
    else:
        values = values.numpy()

    if isinstance(color_map, str):
        color_map = plt.get_cmap(color_map)

    class_colors = get_class_rgba(color_map, values)

    means_subset, covariances_subset = statistics_dim_subset(
        means, covariances, filter_pair
    )

    for _, ind in enumerate(classes_plot):
        single_ellipse(
            center=means_subset[ind],
            covariance=covariances_subset[ind],
            ax=ax,
            color=class_colors[ind],
        )
    ax.autoscale_view()

    ax.set_xlabel(f"Response {filter_pair[0] + 1}")
    ax.set_ylabel(f"Response {filter_pair[1] + 1}")

    if legend_type == 'continuous':
        color_map, norm = get_normalized_color_map(color_map, values)
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, **kwargs)

    elif legend_type == 'discrete':
        for _, ind in enumerate(classes_plot):
            ax.scatter([], [], c=[class_colors[ind]], label=values[ind])
        ax.legend(**kwargs)

    return ax
