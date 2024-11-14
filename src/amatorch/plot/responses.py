import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data_wrangle import subsample_class_points, subsample_classes
from .colors import get_class_rgba, get_normalized_color_map


def scatter_responses(
    responses, labels, ax=None, values=None, filter_pair=(0, 1), n_points=1000,
    classes_plot=None, legend_type='none', **kwargs
):
    """
    Plot scatter of the responses to different categories.

    Parameters
    ----------
    responses : torch.Tensor
        Responses to the stimuli. Shape (n_stimuli, n_filters).
    labels : torch.int64
        Class labels of each point with shape (n_points).
    ax : matplotlib.axes.Axes, optional
        Axes to plot the scatter. If None, a new figure is created.
        The default is None.
    values : torch.Tensor, optional
        Values to color the classes. The default is linearly spaced values
        between -1 and 1.
    filter_pair : tuple, optional
        Pair of filters to plot. The default is (0, 1).
    n_points : int, optional
        Number of points per class to plot. The default is 1000.
    classes_plot : list, optional
        List of classes to plot. The default is all classes.
    legend_type : str, optional
        Type of legend to add: 'none', 'continuous', 'discrete'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the scatter plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Subsample points
    responses_plt, labels_plt = subsample_class_points(responses, labels, n_points)
    responses_plt = responses_plt[:, filter_pair].detach().cpu().numpy()
    labels_plt = labels_plt.detach().cpu().numpy()
    responses_plt, labels_plt = subsample_classes(responses_plt, labels_plt, classes_plot)

    if values is None:
        values = np.arange(len(np.unique(labels_plt)))
    else:
        values = values.numpy()

    color_map = plt.get_cmap("viridis")
    class_colors = get_class_rgba(color_map, values)

    ax.scatter(
        responses_plt[:, 0],
        responses_plt[:, 1],
        c=class_colors[labels_plt],
        s=10,
        alpha=0.5,
    )

    ax.set_xlabel(f"Response {filter_pair[0] + 1}")
    ax.set_ylabel(f"Response {filter_pair[1] + 1}")

    if legend_type == 'continuous':
        color_map, norm = get_normalized_color_map(color_map, values)
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, **kwargs)

    elif legend_type == 'discrete':
        for _, class_ind in enumerate(np.unique(labels)):
            ax.scatter([], [], c=[class_colors[class_ind]], label=values[class_ind])
        ax.legend(**kwargs)

    return ax
