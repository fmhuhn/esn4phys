import os
import h5py

import numpy as np

from .Dataset import Dataset
from .EchoStateNetwork import EchoStateNetwork
from .HybridEchoStateNetwork import HybridEchoStateNetwork
from .TanhHybridEchoStateNetwork import TanhHybridEchoStateNetwork
from .Util import path_and_name

class Case:
    def __init__(self, echo_state_network, dataset, N_train, N_skip, tikh,
            last_state=None, Wout=None, file_path=None):
        """ Create a Case object, which is composed of an echo state network,
            a data set and training parameter.

            Args:
                echo_state_network (EchoStateNetwork): echo state network of
                    the case
                dataset (Dataset): dataset on which to train the echo state
                    network
                N_train (int): number of time steps of dataset to use for
                    training
                N_skip (int): number of time steps to skip due to reservoir
                    transient
                tikh (float): Tikhonov regularization factor
                last_state (np.ndarray): last reservoir state (can be used to
                    resume prediction)
                file_path (str): file path of the json case file
        """

        self.esn = echo_state_network
        self.ds = dataset
        self.N_train = N_train
        self.N_skip = N_skip
        self.tikh = tikh
        self.last_state = last_state
        self.Wout = Wout
        self.file_path = file_path

    def save(self):
        """ Save case file under the path in `self.file_path'. Thus, 
            `self.file_path' must have been set before calling this.
        """

        with h5py.File(self.file_path, 'w') as hf:
            self.esn.save()
            hf.attrs['esn_fp'] = self.esn.file_path if self.esn else None
            hf.attrs['ds_fp'] = self.ds.file_path if self.ds else None
            hf.attrs['N_train'] = self.N_train
            hf.attrs['N_skip'] = self.N_skip
            hf.attrs['tikh'] = self.tikh
            if self.last_state is not None:
                hf.create_dataset('last_state', data=self.last_state)
            if self.Wout is not None:
                hf.create_dataset('Wout', data=self.Wout)

    @staticmethod
    def load(file_path, esn_type='normal'):
        """ Load case.

            Args:
                file_path (str): path of json case file to load
                esn_type (str): type of Echo State Network (normal, hybrid, tanh_hybrid)
        """
        
        if esn_type == 'normal':
            esn_class = EchoStateNetwork
        elif esn_type == 'hybrid':
            esn_class = HybridEchoStateNetwork
        elif esn_type == 'tanh_hybrid':
            esn_class = TanhHybridEchoStateNetwork
        else:
            raise ValueError('Invalid ESN type {}.'.format(esn_type))

        with h5py.File(file_path, 'r') as hf:
            esn = esn_class.load(hf.attrs['esn_fp'])
            ds = Dataset.load(hf.attrs['ds_fp'])
            N_train = hf.attrs['N_train']
            N_skip = hf.attrs['N_skip']
            tikh = hf.attrs['tikh']
            last_state = hf['last_state'][:] if 'last_state' in hf else None
            Wout = hf['Wout'][:] if 'Wout' in hf else None

        return Case(esn, ds, N_train, N_skip, tikh, last_state, Wout,
                file_path)
