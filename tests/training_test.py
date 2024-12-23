##################

# TESTS THAT THE AMA CLASS INITIALIZES AND TRAINS
# FOR THE RELEASE BEFORE REFACTORING
#
##################

import pytest
import torch

import amatorch.optim as optim
from amatorch.datasets import disparity_data
from amatorch.models import AMAGauss

# Set random seed
torch.manual_seed(0)

# Initialize the AMA class
N_EPOCHS = 5
LR = 0.1
LR_STEP = 5
LR_GAMMA = 0.5
BATCH_SIZE = 1024
RESPONSE_NOISE = 0.02
C50 = 0.5


@pytest.fixture(scope="module")
def data():
    return disparity_data()


######## TEST OPTIMIZATION
def test_initialize_filters_pca(data):
    """
    Test that initializing the AMA model filters via PCA yields
    filters whose variances are larger than random variances.
    """
    n_filters = 8
    # Create an AMA model with 8 filters
    ama = AMAGauss(
        stimuli=data["stimuli"],
        labels=data["labels"],
        n_filters=n_filters,
        response_noise=RESPONSE_NOISE,
        c50=C50,
    )

    # Get variances of random filters
    with torch.no_grad():
        responses = ama.get_responses(data["stimuli"])
    variances_rand = torch.var(responses, dim=0)

    # Initialize filters using PCA
    optim.initialize_filters_pca(ama, data["stimuli"])

    # Get variances of PCA filters
    with torch.no_grad():
        responses = ama.get_responses(data["stimuli"])
    variances = torch.var(responses, dim=0)

    # Check that variances of PCA filters are larger than random variances
    assert torch.all(variances > variances_rand), "PCA filters have smaller variance than random filters"
    # also check that no filters are NaN
    assert not torch.isnan(ama.filters).any(), "PCA filters contain NaN"


@pytest.mark.parametrize("n_filters", [2, 4])
@pytest.mark.parametrize("pairwise", [False])
@pytest.mark.parametrize("optimizer", ["LBFGS"])
@pytest.mark.parametrize(
  "scheduler_dict", [{"step_size": LR_STEP, "gamma": LR_GAMMA}]
)
def test_training(data, n_filters, pairwise, optimizer, scheduler_dict):
    ama = AMAGauss(stimuli=data["stimuli"],
        labels=data["labels"],
        n_filters=n_filters,
        response_noise=RESPONSE_NOISE,
        c50=C50,
    )

    optim.initialize_filters_pca(ama, data["stimuli"])

    # Fit model
    loss, training_time = optim.fit(
        model=ama,
        stimuli=data["stimuli"],
        labels=data["labels"],
        max_epochs=N_EPOCHS,
        lr=LR,
        pairwise=pairwise,
        optimizer_name=optimizer,
        batch_size=BATCH_SIZE,
        scheduler_params=scheduler_dict,
        return_loss=True,
        show_progress=False,
    )

    # Get the posteriors
    posteriors = ama.get_posteriors(data["stimuli"][:10])

    # Sample from the distribution
    assert not torch.isnan(ama.filters.detach()).any(), "Filters are nan"
    assert loss[0] > loss[-1], "Loss did not decrease"
    assert not torch.isnan(posteriors).any(), "Posteriors are nan"

