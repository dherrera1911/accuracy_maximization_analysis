import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
import time

## FUNCTIONS FOR FITTING AMA MODELS

# Define loop function to train the model
def fit(nEpochs, model, trainDataLoader, lossFun, opt, scheduler=None):
    """ Fit AMA model using the posterior distribuions generated by the model.
    nEpochs: Number of epochs. Integer
    model: AMA model object.
    trainDataLoader: data loader generated with torch.utils.data.DataLoader
    lossFun: loss function that uses posterior distribution over classes.
    opt: optimizer, selected from torch.optim
    scheduler: scheduler for adaptive learning rate, generated with optim.lr_scheduler
    """
    trainingLoss = np.zeros(nEpochs+1)
    elapsedTime = np.zeros(nEpochs+1)
    # Get the loss of the full dataset stored in the data loader
    trainingLoss[0] = lossFun(model, trainDataLoader.dataset.tensors[0],
            trainDataLoader.dataset.tensors[1]).detach()
    print('Initial loss: ', trainingLoss[0])
    opt.zero_grad()
    # Measure time and start loop
    start = time.time()
    for epoch in range(nEpochs):
        for sb, ctgb in trainDataLoader:
            # Generate predictions for batch sb, returned by trainDataLoader 
            model.update_response_statistics()
            loss = lossFun(model, sb, ctgb)  # Compute loss
            loss.backward()             # Compute gradient
            opt.step()                  # Take one step
            opt.zero_grad()             # Restart gradient
        # Print model loss
        trainingLoss[epoch+1] = lossFun(model, trainDataLoader.dataset.tensors[0],
                trainDataLoader.dataset.tensors[1]).detach()
        trainingDiff = trainingLoss[epoch+1] - trainingLoss[epoch]
        print('Epoch: %d |  Training loss: %.4f  |  Loss change: %.4f' %
                (epoch+1, trainingLoss[epoch+1], trainingDiff))
        end = time.time()
        elapsedTime[epoch+1] = end - start
        # Apply scheduler step
        if not scheduler == None:
            if "ReduceLROnPlateau" in str(type(scheduler)):
                scheduler.step(trainingLoss[epoch+1])    # adapt learning rate
            else:
                scheduler.step()
    # Do the final response statistics update
    model.update_response_statistics()
    return trainingLoss, elapsedTime


# Define loop function to train the model
def fit_by_pairs(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        nPairs, scheduler_fun=None, seedsByPair=1):
    """ Fit AMA model training filters by pairs. After a pair is trained, it is fixed
    in place (no longer trainable), and a new set of trainable filters is then
    initialized and trained. Has the option to try different seeds for each pair of
    filters trained, and choosing the best pair
    nEpochs: Number of epochs for each pair of filters. Integer
    model: AMA model object.
    trainDataLoader: data loader generated with torch.utils.data.DataLoader
    lossFun: loss function that uses posterior distribution over classes.
    opt_fun: A function that takes in a model and returns an optimizer
    nPairs: Number of pairs to train. nPairs=1 corresponds to only training the filters
        included in the input model.
    scheduler_fun: Function that takes in an optimizer and returns a scheduler for
        that optimizer.
    seedsByPair: number of times to train each pair from different random initializations,
        to choose the best pair.
    """
    trainingLoss = [None] * nPairs
    elapsedTimes = [None] * nPairs
    # Measure time and start loop
    start = time.time()
    for p in range(nPairs):
        # If not the first iteration, fix current filters and add new trainable
        if (p>0):
            fAll = model.fixed_and_trainable_filters().detach().clone()
            model.add_fixed_filters(fAll)
            model.reinitialize_trainable()
        print(f'Pair {p}')
        # Train the current pair of trainable filters
        trainingLoss[p], elapsedTimes[p] = fit_multiple_seeds(nEpochs=nEpochs,
                model=model, trainDataLoader=trainDataLoader, lossFun=lossFun,
                opt_fun=opt_fun, scheduler_fun=scheduler_fun, nSeeds=seedsByPair)
        end = time.time()
        elapsedTime = end - start
        print(f'########## Pair {p+1} trained in {elapsedTime} ##########')
    # Put all the filters into the f model attribute
    fAll = model.fixed_and_trainable_filters().detach().clone()
    model.assign_filter_values(fAll)
    model.add_fixed_filters(torch.tensor([]))
    return trainingLoss, elapsedTimes


# LOOP TO TRAIN MULTIPLE SEEDS AND CHOOSE BEST
def fit_multiple_seeds(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        scheduler_fun=None, nSeeds=1):
    """ Fit AMA model multiple times from different seeds, and keep the result with
    best performance. 
    nEpochs: Number of epochs for each pair of filters. Integer
    model: AMA model object.
    trainDataLoader: data loader generated with torch.utils.data.DataLoader
    lossFun: loss function that uses posterior distribution over classes.
    opt_fun: A function that takes in a model and returns an optimizer
    scheduler_fun: Function that takes in an optimizer and returns a scheduler for
        that optimizer.
    nSeeds: number of times to train the filters among which to choose the best ones
    """
    if (nSeeds>1):
        # Initialize lists to fill
        seedLoss = np.zeros(nSeeds)
        trainingLoss = [None] * nSeeds
        elapsedTimes = [None] * nSeeds
        filters = [None] * nSeeds
        for p in range(nSeeds):
            if (p>0):
                model.reinitialize_trainable()
            # Train the current pair of trainable filters
            opt = opt_fun(model)
            if (scheduler_fun == None):
                scheduler = None
            else:
                scheduler = scheduler_fun(opt)
            # Train random initialization of the model
            trainingLoss[p], elapsedTimes[p] = fit(nEpochs=nEpochs, model=model,
                    trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
                    scheduler=scheduler)
            filters[p] = model.f.detach().clone()
            # Get the final loss of the filters of this seetrainingLossd
            seedLoss[p] = trainingLoss[p][-1]
        # Set the filter with the minimum loss into the model
        minFilt = seedLoss.argmin()
        model.assign_filter_values(filters[minFilt])
        # Return only the training loss history of the best filter
        minLoss = trainingLoss[minFilt]
        minElapsed = elapsedTimes[minFilt]
    else:
        opt = opt_fun(model)
        if (scheduler_fun == None):
            scheduler = None
        else:
            scheduler = scheduler_fun(opt)
        minLoss, minElapsed = fit(nEpochs=nEpochs, model=model,
                trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
                scheduler=scheduler)
    return minLoss, minElapsed

## LOSS FUNCTIONS
# Define loss functions that take as input AMA model, so
# different outputs can be used with the same fitting functions


def cross_entropy_loss():
    """ Cross entropy loss for AMA.
    model: AMA model object
    s: input stimuli. tensor shaped batch x features
    ctgInd: true categories of stimuli, as a vector with category index
        type torch.LongTensor"""
    crossEnt = torch.nn.CrossEntropyLoss()
    def lossFun(model, s, ctgInd):
        loss = crossEnt(model.get_posteriors(s), ctgInd)
        return loss
    return lossFun

def mse_loss():
    """ MSE loss for AMA. Computes MSE between the latent variable
    estimate 
    model: AMA model object
    s: input stimuli. tensor shaped batch x features
    ctgInd: true categories of stimuli. type torch.LongTensor"""
    mseLoss = torch.nn.MSELoss()
    def lossFun(model, s, ctgInd):
        loss = mseLoss(model.get_estimates(s, method4est='MMSE'),
                model.ctgVal[ctgInd])
        return loss
    return lossFun

def mae_loss():
    """ MAE loss for AMA. Computes MAE between the latent variable
    estimate 
    model: AMA model object
    s: input stimuli. tensor shaped batch x features
    ctgInd: true categories of stimuli. type torch.LongTensor"""
    mseLoss = torch.nn.L1Loss()
    def lossFun(model, s, ctgInd):
        loss = mseLoss(model.get_estimates(s, method4est='MMSE'),
                model.ctgVal[ctgInd])
        return loss
    return lossFun

## FUNCTIONS FOR SUMMARY AND EVALUATION OF MODEL PERFORMANCE
# Function that turns posteriors into estimate averages, SDs and CIs
def get_estimate_statistics(estimates, ctgInd, quantiles=[0.05, 0.95]):
    # Compute means and stds for each true level of the latent variable
    estimatesMeans = torch.zeros(ctgInd.max()+1)
    estimatesSD = torch.zeros(ctgInd.max()+1)
    lowCI = torch.zeros(ctgInd.max()+1)
    highCI = torch.zeros(ctgInd.max()+1)
    quantiles = torch.tensor(quantiles, dtype=torch.float64)
    for cl in ctgInd.unique():
        levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
        estimatesMeans[cl] = estimates[levelInd].mean()
        estimatesSD[cl] = estimates[levelInd].std()
        (lowCI[cl], highCI[cl]) = torch.quantile(estimates[levelInd], quantiles)
    return {'estimateMean': estimates, 'estimateSD': estimatesSD,
            'lowCI': lowCI, 'highCI': highCI}


## FUNCTIONS FOR PLOTTING FILTERS OF AMA MODEL
# Functions for plotting the filters
def unvectorize_filter(fIn, frames=15, pixels=30):
    nFilt = fIn.shape[0]
    matFilt = fIn.reshape(nFilt, 2, frames, pixels)
    matFilt = matFilt.transpose(1,2).reshape(nFilt, frames, pixels*2)
    return matFilt

# Plot filters
def view_filters_bino_video(fIn, frames=15, pixels=30):
    matFilt = unvectorize_filter(fIn, frames=frames, pixels=pixels)
    nFilters = matFilt.shape[0]
    for k in range(nFilters):
        plt.subplot(nFilters, 1, k+1)
        plt.imshow(matFilt[k,:,:].squeeze())
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

# DEFINE A FUNCTION TO VISUALIZE BINOCULAR FILTERS
def view_filters_bino(f, x=[], title=''):
    plt.title(title)
    nPixels = int(max(f.shape)/2)
    if len(x) == 0:
        x = np.arange(nPixels)
    plt.plot(x, f[:nPixels], label='L', color='red')
    plt.plot(x, f[nPixels:], label='R', color='blue')
    plt.ylim(-0.3, 0.3)

# DEFINE A FUNCTION TO VISUALIZE ALL BINOCULAR FILTERS OF AN AMA MODEL
def view_all_filters_bino(amaPy, x=[]):
    fAll = amaPy.fixed_and_trainable_filters()
    fAll = fAll.detach()
    nFiltAll = fAll.shape[0]
    nPairs = int(nFiltAll/2)
    for n in range(nFiltAll):
        plt.subplot(nPairs, 2, n+1)
        view_filters_bino(fAll[n,:], x=[], title=f'F{n}')


