import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import pycircstat as pcirc
import time


##################################
##################################
#
## FUNCTIONS FOR FITTING AMA MODELS
#
##################################
##################################
#
# This group of functions take an ama model, and some inputs
# such as the loss function, and do the training loop.
# Different types of training are available, such as training
# the filters in pairs, or from multiple seeds


# Define loop function to train the model
def fit(nEpochs, model, trainDataLoader, lossFun, opt, scheduler=None,
        sTst=None, ctgIndTst=None, printProg=True):
    """
    Fit AMA model using the posterior distribuions generated by the model.
    ----------------
    Arguments:
    ----------------
      - nEpochs: Number of epochs. Integer.
      - model: AMA model object.
      - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
      - lossFun: Loss function to evaluate.
      - opt: Optimizer, selected from torch.optim.
      - scheduler: Scheduler for adaptive learning rate, generated with
              optim.lr_scheduler. Default is None.
      - sTst: Test stimulus matrix, used for computing test
              loss. (nStim x nDim). Default is None.
      - ctgIndTst: Vector indicating category of each test stimulus row.
              Used for computing test loss. (nStim). Default is None.
    ----------------
    Outputs:
    ----------------
      - trnLoss: Vector of training loss at each epoch. (nEpochs)
      - tstLoss: Vector of test loss at each epoch
      - elapsedTime: Vector of elapsed time at each epoch. (nEpochs)
    """
    trnLoss = np.zeros(nEpochs+1)
    tstLoss = np.zeros(nEpochs+1)
    elapsedTime = np.zeros(nEpochs+1)
    # Get the loss of the full dataset stored in the data loader
    trnLoss[0] = lossFun(model=model, s=trainDataLoader.dataset.tensors[0],
                              ctgInd=trainDataLoader.dataset.tensors[1]).detach()
    if not sTst == None:
        tstLoss[0] = lossFun(model=model, s=sTst, ctgInd=ctgIndTst)
    print(f"Init Train loss: {trnLoss[0]:.4f}  | "
          f"Test loss: {tstLoss[0]:.4f}")
    opt.zero_grad()
    # TAKE THE TIME AND START LOOP
    start = time.time()
    if printProg:
        # Print headers
        print("-"*72)
        print(f"{'Epoch':^5} | {'Train loss':^12} | {'Diff (e-3)':^10} | "
              f"{'Test loss':^12} | {'Diff (e-3)':^10} | {'Time (s)':^8}")
        print("-"*72)
    for epoch in range(nEpochs):
        ### MAIN TRAINING LOOP
        for sb, ctgb in trainDataLoader:
            # Update model statistics to the new filters
            model.update_response_statistics()
            loss = lossFun(model, sb, ctgb)     # Compute loss
            loss.backward()                     # Compute gradient
            opt.step()                          # Take one step
            opt.zero_grad()                     # Restart gradient
        ### PRINT MODEL LOSS
        trnLoss[epoch+1] = lossFun(
            model=model,
            s=trainDataLoader.dataset.tensors[0],
            ctgInd=trainDataLoader.dataset.tensors[1]).detach()
        trainingDiff = trnLoss[epoch+1] - trnLoss[epoch]
        if not sTst == None:
            tstLoss[epoch+1] = lossFun(model=model, s=sTst, ctgInd=ctgIndTst)
        tstDiff = tstLoss[epoch+1] - tstLoss[epoch]
        # Print progress
        if printProg:
            print(f"{epoch+1:^5} | "
                  f"{trnLoss[epoch]:>12.3f} | "
                  f"{trainingDiff*1000:>10.1f} | "
                  f"{tstLoss[epoch]:>12.3f} | "
                  f"{tstDiff*1000:>10.1f} | "
                  f"{elapsedTime[epoch]:^8.1f}")
        end = time.time()
        elapsedTime[epoch+1] = end - start
        # Apply scheduler step
        if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(trnLoss[epoch+1])    # adapt learning rate
        elif type(scheduler) == optim.lr_scheduler.StepLR:
            scheduler.step()
    print("")
    # DO THE FINAL RESPONSE STATISTICS UPDATE
    model.update_response_statistics()
    return trnLoss, tstLoss, elapsedTime


# LOOP TO TRAIN MULTIPLE SEEDS AND CHOOSE BEST
def fit_multiple_seeds(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        nSeeds=1, scheduler_fun=None, sTst=None, ctgIndTst=None,
        printProg=False):
    """
    Fit AMA model multiple times from different seeds, and keep the result with
    best performance.
    ----------------
    Arguments:
    ----------------
      - nEpochs: Number of epochs for each pair of filters. Integer.
      - model: AMA model object.
      - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
      - lossFun: Loss function that uses posterior distribution over classes.
      - opt_fun: A function that takes in a model and returns an optimizer.
      - nSeeds: Number of times to train the filters among which to choose
              the best ones. Default is 1.
      - scheduler_fun: Function that takes in an optimizer and returns
              a scheduler for that optimizer. Default is None.
      - sTst: Test stimulus matrix, used for computing test
              loss. (nStim x nDim). Default is None.
      - ctgIndTst: Vector indicating category of each test stimulus row.
              Used for computing test loss. (nStim). Default is None.
    ----------------
    Outputs:
    ----------------
      - trnLoss: Numpy array with training loss for each seed, with rows
              sorted in increasing order of final loss (nSeeds x nEpochs).
      - tstLoss: Numpy array with test loss for each seed, with rows
              sorted in increasing order of final loss (nSeeds x nEpochs).
      - elapsedTime: Numpy array with elapsed time for each seed, with
              rows sorted in increasing order of final loss (nSeeds x nEpochs).
      - filters: List of filters for each seed, sorted in increasing
              order of final loss (nSeeds x nDim).
    """
    # INITIALIZE LISTS TO FILL WITH TRAINING PROGRESS INFORMATION
    seedLoss = np.zeros(nSeeds)
    trnLoss = np.zeros((nSeeds, nEpochs+1))
    tstLoss = np.zeros((nSeeds, nEpochs+1))
    elapsedTimes = np.zeros((nSeeds, nEpochs+1))
    filters = [None] * nSeeds
    # LOOP OVER SEEDS
    for p in range(nSeeds):
        print(f'##########      SEED {p+1}      ########## \n ')
        # If not first seed, reinitialize the model
        if (p>0):
            model.reinitialize_trainable()
            model.update_response_statistics()
        # Set up optimizer and scheduler
        opt = opt_fun(model)
        if (scheduler_fun == None):
            scheduler = None
        else:
            scheduler = scheduler_fun(opt)
        # TRAIN MODEL WITH THIS SEED
        trnLoss[p,:], tstLoss[p,:], elapsedTimes[p,:] = fit(
            nEpochs=nEpochs, model=model, trainDataLoader=trainDataLoader,
            lossFun=lossFun, opt=opt, scheduler=scheduler, sTst=sTst,
            ctgIndTst=ctgIndTst, printProg=printProg)
        # Save filters
        filters[p] = model.f.detach().clone()
        # Get the loss for these filters
        if not sTst is None:
            seedLoss[p] = tstLoss[p,-1]
        else:
            seedLoss[p] = trnLoss[p,-1]
    # Put best filter into the model
    minFilt = seedLoss.argmin()
    model.assign_filter_values(fNew=filters[minFilt])
    model.update_response_statistics()
    # Sort outputs by increasing loss
    trnLoss = trnLoss[seedLoss.argsort(),:]
    tstLoss = tstLoss[seedLoss.argsort(),:]
    elapsedTimes = elapsedTimes[seedLoss.argsort(),:]
    filters = [filters[i] for i in seedLoss.argsort()]
    return trnLoss, tstLoss, elapsedTimes, filters


# TRAIN MODEL FILTERS IN PAIRS, WITH POSSIBLE SEED SELECTION
def fit_by_pairs(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        nPairs, scheduler_fun=None, seedsByPair=1, sTst=None, ctgIndTst=None,
        printProg=False):
    """
    Fit AMA model training filters by pairs. After a pair is trained, it
    is fixed in place (no longer trainable), and a new set of trainable
    filters is then initialized and trained. Has the option to try different
    seeds for each pair of filters trained, and choosing the best pair
    ----------------
    Arguments:
    ----------------
      - nEpochs: Number of epochs for each pair of filters. Integer.
      - model: AMA model object.
      - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
      - lossFun: Loss function that uses posterior distribution over classes.
      - opt_fun: A function that takes in a model and returns an optimizer.
      - nPairs: Number of pairs to train. nPairs=1 corresponds to only training
          the filters included in the input model.
      - seedsByPair: Number of times to train each pair from different random
          initializations, to choose the best pair. Default is 1.
      - scheduler_fun: Function that takes in an optimizer and returns
              a scheduler for that optimizer. Default is None.
      - sTst: Test stimulus matrix, used for computing test
              loss. (nStim x nDim). Default is None.
      - ctgIndTst: Vector indicating category of each test stimulus row.
              Used for computing test loss. (nStim). Default is None.
    ----------------
    Outputs:
    ----------------
      - trnLoss: List with training loss for each pair of filters trained.
              list of length nPairs, each element is a tensor with size
              (seedsByPair x nEpochs)
      - tstLoss: List with test loss for each pair of filters trained.
              list of length nPairs, each element is a tensor with size
              (seedsByPair x nEpochs)
      - elapsedTimes: List with elapsed times for each pair of filters trained.
              list of length nPairs, each element is a tensor with size
              (seedsByPair x nEpochs)
      - filters: List of different seed filters for each pair trained.
              list of length nPairs, where each element is a list of length
              seedByPair, containing a tensor with the filters trained at
              that step, of size (2 x nDim)
    """
    trnLoss = [None] * nPairs
    tstLoss = [None] * nPairs
    elapsedTimes = [None] * nPairs
    filters = [None] * nPairs
    # Measure time and start loop
    start = time.time()
    for p in range(nPairs):
        # If not the first iteration, fix current filters and add new trainable
        if (p>0):
            model.move_trainable_2_fixed()
            model.update_response_statistics()
        print("#"*45)
        print(f'##########      FILTER PAIR {p+1}      ##########')
        print("#"*45, "\n ")
        # Train the current pair of trainable filters
        trnLoss[p], tstLoss[p], elapsedTimes[p], filters[p] = \
                fit_multiple_seeds(
                    nEpochs=nEpochs, model=model, trainDataLoader=trainDataLoader,
                    lossFun=lossFun, opt_fun=opt_fun, nSeeds=seedsByPair,
                    scheduler_fun=scheduler_fun, sTst=sTst, ctgIndTst=ctgIndTst,
                    printProg=printProg)
        end = time.time()
        elapsedTime = end - start
        minutes, seconds = divmod(int(elapsedTime), 60)
        print(f'########## PAIR {p+1} TRAINED IN {minutes:02d}:{seconds:02d} '
              '########## \n ')
    # Put all the filters into f
    fAll = model.all_filters().detach().clone()
    model.assign_filter_values(fAll)
    model.add_fixed_filters(fFixed=torch.tensor([]))
    model.update_response_statistics()
    return trnLoss, tstLoss, elapsedTimes, filters

