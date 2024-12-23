"""
Routine to fit AMA filters using Gradient Descent.
"""

import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from tqdm import tqdm

from amatorch.constraints import FixedFilters

__all__ = ["fit", "initialize_filters_pca"]


def __dir__():
    return __all__


def kl_loss(model, stimuli, labels):
    """
    Compute the negative log-likelihood loss (KL loss) for the AMA model.

    Parameters
    ----------
    model : AMA model object
        The model used for loss computation.
    stimuli : torch.Tensor
        Input stimuli tensor of shape (batch_size, n_channels, n_dim).
    labels : torch.Tensor
        True category labels for the stimuli (vector of category indices).

    Returns
    -------
    torch.Tensor
        Negative log-likelihood loss.
    """
    n_stimuli = stimuli.shape[0]
    log_posteriors = torch.log(model.get_posteriors(stimuli) + 1e-8)
    correct_log_posteriors = log_posteriors[torch.arange(n_stimuli), labels]
    return -torch.mean(correct_log_posteriors)


def fit(
    model,
    stimuli,
    labels,
    max_epochs=200,
    loss_fun=None,
    lr=0.1,
    atol=1e-6,
    show_progress=True,
    return_loss=False,
    pairwise=False,
    optimizer_name="LBFGS",
    scheduler_params=None,
    batch_size=512,
    **kwargs,
):
    """
    Learn AMA filters using either LBFGS or NAdam, with the option to
    train in pairs (pairwise).

    Parameters
    ----------
    model : AMA model object
    stimuli : torch.Tensor
    labels : torch.Tensor
    max_epochs : int
    loss_fun : callable, optional
    lr : float, optional
    atol : float, optional
    show_progress : bool, optional
    return_loss : bool, optional
    pairwise : bool, optional
    optimizer_name : str, optional
        Which optimizer to use: 'LBFGS' or 'NADAM'.
    scheduler_params : dict or None
        Optional parameters for the scheduler (only used by NAdam).
    batch_size : int, optional
        Batch size for NAdam training.
    **kwargs : dict
        Additional kwargs that might be passed to the chosen fitting loop:
         - For LBFGS, could be { history_size, line_search_fn, max_iter, ... }
         - For NAdam, you might not need these, but you can pass them anyway.

    Returns
    -------
    loss, training_time : torch.Tensor, torch.Tensor
    """
    # Decide which loop function to use
    if optimizer_name.upper() == "LBFGS":
        fitting_function = fitting_loop_lbfgs
    elif optimizer_name.upper() == "NADAM":
        fitting_function = fitting_loop_nadam
    else:
        raise ValueError("optimizer_name must be either 'LBFGS' or 'NADAM'.")

    # If not pairwise, just call the chosen fitting_function once
    if not pairwise:
        # NOTE: For LBFGS, pass `max_epochs=max_epochs, lr=lr, atol=atol, ...`
        #       For NAdam, pass `max_epochs=max_epochs, lr=lr, batch_size=..., scheduler_params=...`
        # You can unify or separate them as you wish.
        if optimizer_name.upper() == "LBFGS":
            loss, training_time = fitting_function(
                model=model,
                stimuli=stimuli,
                labels=labels,
                max_epochs=max_epochs,
                loss_fun=loss_fun,
                lr=lr,
                atol=atol,
                show_progress=show_progress,
                return_loss=True,
                **kwargs,  # Additional LBFGS-specific args
            )
        else:  # NAdam
            loss, training_time = fitting_function(
                model=model,
                stimuli=stimuli,
                labels=labels,
                max_epochs=max_epochs,
                loss_fun=loss_fun,
                lr=lr,
                batch_size=batch_size,
                scheduler_params=scheduler_params,
                show_progress=show_progress,
                return_loss=True,
            )

    else:
        # Pairwise training logic
        initial_filters = model.filters.detach().clone()
        n_filters = model.filters.shape[0]
        if n_filters % 2 != 0:
            raise ValueError("Number of filters must be even for pairwise training.")
        n_pairs = n_filters // 2

        # Start with just the first two filters
        remove_parametrizations(model, "filters")
        model.filters = nn.Parameter(initial_filters[:2])
        model._add_constraint(model.constraint)

        loss = torch.tensor([])
        training_time = torch.tensor([])

        for i in range(n_pairs):
            if i > 0:
                # Add next pair of filters
                next_pair = initial_filters[2*i : 2*(i+1)]
                remove_parametrizations(model, "filters")
                model.filters = nn.Parameter(
                    torch.cat([model.filters, next_pair], dim=0)
                )
                model._add_constraint(model.constraint)
                # Fix filters from previous pairs
                register_parametrization(
                    model, "filters", FixedFilters(n_row_fixed=i*2)
                )

            # Fit the model with current pair(s)
            if optimizer_name.upper() == "LBFGS":
                current_loss, current_time = fitting_loop_lbfgs(
                    model=model,
                    stimuli=stimuli,
                    labels=labels,
                    max_epochs=max_epochs,
                    loss_fun=loss_fun,
                    lr=lr,
                    atol=atol,
                    show_progress=show_progress,
                    return_loss=True,  # we want to collect losses/time
                    **kwargs,
                )
            else:  # NAdam
                current_loss, current_time = fitting_loop_nadam(
                    model=model,
                    stimuli=stimuli,
                    labels=labels,
                    max_epochs=max_epochs,
                    loss_fun=loss_fun,
                    lr=lr,
                    batch_size=batch_size,
                    scheduler_params=scheduler_params,
                    show_progress=show_progress,
                    return_loss=True,
                )

            # Concatenate losses and adjust time to keep running total
            loss = torch.cat([loss, current_loss])
            if i > 0:
                # shift current_time by the last entry in training_time so time is cumulative
                current_time = current_time + training_time[-1]
            training_time = torch.cat([training_time, current_time])

        # Clean up parametrizations
        remove_parametrizations(model, "filters")
        model._add_constraint(model.constraint)

    if return_loss:
        return loss, training_time
    else:
        return None


def fitting_loop_nadam(
    model,
    stimuli,
    labels,
    max_epochs=200,
    loss_fun=None,
    lr=0.01,
    batch_size=512,
    scheduler_params=None,
    show_progress=True,
    return_loss=False,
):
    """
    Example fitting loop for the AMA model using NAdam with mini-batches.

    Parameters
    ----------
    model : AMA model object
    stimuli : torch.Tensor
    labels : torch.Tensor
    max_epochs : int
    loss_fun : callable or None
    lr : float
    batch_size : int
    scheduler_params : dict or None
        e.g. { 'step_size': 1000, 'gamma': 0.95 }
    show_progress : bool
    return_loss : bool

    Returns
    -------
    (loss, training_time) or None
    """
    if loss_fun is None:
        loss_fun = kl_loss  # your existing kl_loss function

    # 1) Create data loader for mini-batch training
    dataset = TensorDataset(stimuli, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(data_loader)

    # 2) Create the NAdam optimizer
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)

    # 3) Optional scheduler
    scheduler = None
    if scheduler_params is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    # 4) Track loss and time
    loss_list = []
    training_time = []
    total_start_time = time.time()

    prev_loss = None  # We can track change if desired

    # 5) Main training loop
    epoch_range = range(max_epochs)
    if show_progress:
        epoch_range = tqdm(epoch_range, desc="Epochs (NAdam)", unit="epoch")

    for e in epoch_range:
        epoch_start_time = time.time()
        running_loss = 0.0

        # Mini-batch training
        for batch_stimuli, batch_labels in data_loader:
            optimizer.zero_grad()
            batch_loss = loss_fun(model, batch_stimuli, batch_labels)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.detach().item()

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Compute average loss, track time
        epoch_loss = running_loss / n_batches
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time

        loss_list.append(epoch_loss)
        training_time.append(epoch_time)

        # (Optional) print or check stopping criteria...
        loss_change = 0.0 if prev_loss is None else abs(epoch_loss - prev_loss)
        prev_loss = epoch_loss

        if show_progress:
            tqdm.write(
                f"Epoch {e+1}/{max_epochs}, Loss: {epoch_loss:.4f}, "
                + f"Change: {loss_change:.4f}, Time: {total_time:.2f}s"
            )

    if return_loss:
        return torch.tensor(loss_list), torch.tensor(training_time)
    else:
        return None


def fitting_loop_lbfgs(
    model,
    stimuli,
    labels,
    max_epochs=200,
    loss_fun=None,
    lr=0.1,
    atol=1e-6,
    show_progress=True,
    return_loss=False,
    **kwargs
):
    """
    Full-batch fitting loop for the AMA model using LBFGS.
    """
    if loss_fun is None:
        loss_fun = kl_loss  # your existing kl_loss function

    # 1) Set up the LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        **kwargs  # e.g., history_size, line_search_fn, max_iter, etc.
    )

    # 2) Tracking variables
    loss_list = []
    training_time = []
    total_start_time = time.time()

    prev_loss = 0.0
    consecutive_stopping_criteria_met = 0

    # 3) Closure used by LBFGS
    def closure():
        optimizer.zero_grad()
        current_loss = loss_fun(model, stimuli, labels)
        current_loss.backward()
        return current_loss

    # 4) Training loop
    epoch_range = range(max_epochs)
    if show_progress:
        epoch_range = tqdm(epoch_range, desc="Epochs (LBFGS)", unit="epoch")

    for e in epoch_range:
        epoch_loss_tensor = optimizer.step(closure)
        epoch_loss = epoch_loss_tensor.item()

        epoch_time = time.time() - total_start_time
        loss_change = abs(prev_loss - epoch_loss)

        if loss_change < atol:
            consecutive_stopping_criteria_met += 1
        else:
            consecutive_stopping_criteria_met = 0

        prev_loss = epoch_loss
        loss_list.append(epoch_loss)
        training_time.append(epoch_time)

        if consecutive_stopping_criteria_met >= 3:
            if show_progress:
                tqdm.write(
                    f"LBFGS: Loss change below {atol} for 3 consecutive epochs. "
                    f"Stopping training at epoch {e + 1}/{max_epochs}."
                )
            break
    else:
        print(
            f"LBFGS: Reached max_epochs ({max_epochs}) without meeting stopping criteria.\n"
            "Consider increasing max_epochs or adjusting hyperparameters."
        )

    if return_loss:
        return torch.tensor(loss_list), torch.tensor(training_time)
    else:
        return None


def initialize_filters_pca(model, stimuli):
    """
    Initialize the model's filters via PCA on the provided `stimuli`.

    Parameters
    ----------
    model : AMA model object
        The model whose filters we want to initialize.
        Expects `model.filters` to be an nn.Parameter of shape (n_filters, n_dim).
    stimuli : torch.Tensor
        Input data of shape (n_samples, n_dim).

    Raises
    ------
    ValueError
        If stimuli is None.
        If number of filters exceeds data dimension.
    """
    if stimuli is None:
        raise ValueError("`stimuli` must be provided to initialize PCA filters.")

    # Check shapes: model.filters.shape -> (n_filters, n_dim)
    n_filters, n_channels, data_dim = model.filters.shape
    if n_filters > data_dim:
        raise ValueError(
            "Number of filters must be less than or equal to the data dimension."
        )

    # Collapse the channel dimension
    stimuli_collapsed = stimuli.view(stimuli.shape[0], -1)  # shape: (n_samples, n_dim)

    # Compute PCA filters
    pca_filters = pca(stimuli_collapsed, n_filters)  # shape: (n_filters, n_dim)

    # Add channel dimension to filters
    pca_filters = pca_filters.view(n_filters, n_channels, data_dim)

    # Remove any parametrizations on "filters"
    remove_parametrizations(model, "filters", leave_parametrized=False)

    # Assign new filters
    model.filters = nn.Parameter(pca_filters)

    # Re-add the model's constraint
    model._add_constraint(constraint=model.constraint)


def pca(points, n_components=None):
    """
    Compute the principal components of the given points.

    Parameters
    ----------
    points : torch.Tensor
        Data points with shape (n_points, n_dim).
    n_components : int
        Number of principal components to compute. Default is
        min(n_points, n_dim).

    Returns
    -------
    components : torch.Tensor
        Principal components of the points. (n_components, n_dim)
    """
    n_points, n_dim = points.shape

    if n_components is None:
        n_components = min(n_points, n_dim)

    if n_components > n_dim:
        raise ValueError("n_components must be less than or equal to n_dim.")

    covariance = torch.matmul(points.T, points) / n_points

    # Compute the eigendecomposition of the covariance matrix
    eigenvalues, components = torch.linalg.eigh(covariance)

    components = components[:, -n_components:]
    components = torch.flip(components, dims=[1]).T

    return components
