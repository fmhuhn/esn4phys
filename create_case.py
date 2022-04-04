import os
import sys
import argparse

import numpy as np

from base.Case import Case
from base.Dataset import Dataset
from base.EchoStateNetwork import EchoStateNetwork
from base.HybridEchoStateNetwork import HybridEchoStateNetwork

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a case.')

    parser.add_argument('echo_state_network', metavar='esn_file_path', type=str,
            help='echo state network file')

    parser.add_argument('dataset', metavar='ds_file_path', type=str,
            help='dataset file')

    parser.add_argument('N_train', metavar='N_train', type=int,
            help='number of time steps for training')

    parser.add_argument('N_skip', metavar='N_skip', type=int,
            help='number of transient time steps in ESN')

    parser.add_argument('tikh', metavar='tikh', type=float,
            help='Tikhonov regularization factor')

    parser.add_argument('file_path', metavar='file_path', type=str,
            help='file path of case file')

    args = parser.parse_args()

    if args.echo_state_network.endswith('hesn'):
        echo_state_network = HybridEchoStateNetwork.load(args.echo_state_network)
    else:
        echo_state_network = EchoStateNetwork.load(args.echo_state_network) if \
            args.echo_state_network else None

    dataset = Dataset.load(args.dataset) if args.dataset else None

    case = Case(echo_state_network, dataset, args.N_train, args.N_skip,
        args.tikh, None, None, args.file_path)

    case.save()
