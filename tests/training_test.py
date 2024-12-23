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

# Initialize the AMA class
N_EPOCHS = 10
LR = 0.1
LR_STEP = 5
LR_GAMMA = 0.5
BATCH_SIZE = 1024
RESPONSE_NOISE = 0.01
C50 = 0.3


@pytest.fixture(scope="module")
def data():
    return disparity_data()


######## TEST THAT AMA RUNS ########
@pytest.mark.parametrize("pairwise", [True, False])
def test_training(data, pairwise):
    ama = AMAGauss(stimuli=data["stimuli"],
        labels=data["labels"],
        n_filters=6,
        response_noise=RESPONSE_NOISE,
        c50=C50,
    )

    # Fit model
    loss, training_time = optim.fit(
        model=ama,
        stimuli=data["stimuli"],
        labels=data["labels"],
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        decay_step=LR_STEP,
        decay_rate=LR_GAMMA,
        pairwise=pairwise,
    )

    # Get the posteriors
    posteriors = ama.get_posteriors(data["stimuli"])

    # Sample from the distribution
    assert not torch.isnan(ama.filters.detach()).any(), "Filters are nan"
    assert loss[0] > loss[-1], "Loss did not decrease"
    assert not torch.isnan(posteriors).any(), "Posteriors are nan"
