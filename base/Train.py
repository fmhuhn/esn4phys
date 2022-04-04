import os
import h5py

import numpy as np

def split_timeseries(timeseries, split_lengths, overlap=None, rest=False):
    """ Splits the time series into intervals. For example, split between
        training, validation and test sets.

        Args:
            timeseries (np.ndarray): vector of timeseries (time in first axis)
            split_lengths (nd.array): vector of lengths of the splits
            overlap (nd.array): an array with the number of overlap time steps
                between consecutive intervals
            rest (bool): whether the function should return what is left after
                splitting according to split_lengths (if len(timeseries) <
                sum(split_lengths))
    """
    if not overlap:
        overlap = [0]*len(split_lengths)
    
    splits = []
    idx = 0
    for length, overlap_add in zip(split_lengths, overlap): 
        splits.append(timeseries[idx:idx+length+overlap_add])
        idx += length

    if rest:
        splits.append(timeseries[idx:])

    return splits

def train_network(esn, U_skip, U_train, Y_train, xa0, tikh, reg_noise=0.0):
    """ Train network with ridge regression.

        Args:
            esn (EchoStateNetwork): network to be trained
            U_skip (np.ndarray): ESN transient set
            U_train (nd.ndarray): training input
            Y_train (nd.ndarray): training target
            xa0 (nd.ndarray): initial ESN state
            tikh (float): Tikhonov parameter
            reg_noise (float): regularizing noise
    """
    if U_skip is None:
        Xa_skip = xa0.reshape((1,-1))
    else:
        Xa_skip, _ = esn.open_loop(U_skip, xa0)

    Xa_train, _ = esn.open_loop(U_train, Xa_skip[-1])

    # Calculate optimal Wout
    if reg_noise > 0:
        Xa = Xa_train[1:].copy()
        Xa += reg_noise*np.random.normal(size=Xa.shape)
    else:
        Xa = Xa_train[1:]

    Y = Y_train
    LHS = np.dot(Xa.T, Xa) + tikh*np.eye(esn.xa_dim)
    RHS = np.dot(Xa.T, Y)
    Wout = np.linalg.solve(LHS, RHS)

    return Wout, Xa
