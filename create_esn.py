import os
import sys
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an Echo State Network.')

    parser.add_argument('N_units', metavar='N_units', type=int,
        help='number of units in reservoir')

    parser.add_argument('N_dim', metavar='N_dim', type=int,
        help='dimension of input/output')

    parser.add_argument('rho', metavar='rho', type=float,
        help='spectral radius of W')

    parser.add_argument('sigma_in', metavar='sigma_in', type=float,
        help='uniform distribution range of Win')

    parser.add_argument('--sparseness', metavar='sp', type=float,
        help='sparseness of W (fraction)', default=None)

    # parser.add_argument('--activation_name', metavar='activ', type=str,
    #     help='activation function', default='tanh')

    parser.add_argument('--bias_in', metavar='b_in', type=float,
        help='input bias', default=None)

    parser.add_argument('--bias_out', metavar='b_out', type=float,
        help='output bias', default=None)

    parser.add_argument('--rand_seed', metavar='rand_seed', type=int,
        help='random seed', default=None)

    parser.add_argument('file_path', metavar='file_path', type=str,
        help='file path of case file')

    parser.add_argument('--phys', metavar='system', type=str,
        help='file path of dynamical system file (hybrid ESN)', default=None)

    parser.add_argument('--Gamma', metavar='Gamma', type=float,
        help='fraction of nodes whose input is from the physical model (hESN)',
        default=0.5)

#    parser.add_argument('--tanh_scale', metavar='tanh_scale', type=float,
#        help='scaling factor inside tanh of physics output',
#        default=1.0)

    parser.add_argument('--esn_type', metavar='esn_type', type=str,
        help='type of echo state network (normal, hybrid, tanh_hybrid',
        default='normal')

    args = parser.parse_args()
    args.activation_name = 'tanh'

    rand = np.random.RandomState(args.rand_seed)

    # TODO: clean up duplicated code
    if args.esn_type == 'normal':
        # Conventional ESN
        from base.EchoStateNetwork import EchoStateNetwork

        esn = EchoStateNetwork(args.N_units, args.N_dim, args.rho,
                args.sparseness, args.sigma_in, args.activation_name,
                args.bias_in, args.bias_out, None, None, rand, args.file_path)
    
    elif args.esn_type == 'hybrid':
        from base.DynamicalSystems import DynamicalSystem
        from base.HybridEchoStateNetwork import HybridEchoStateNetwork
        
        phys = DynamicalSystem.load(args.phys)
        
        esn = HybridEchoStateNetwork(phys, args.Gamma, 
                args.N_units, args.N_dim, args.rho, args.sparseness,
                args.sigma_in, args.activation_name, args.bias_in,
                args.bias_out, None, None, rand, args.file_path)

#    elif args.esn_type == 'tanh_hybrid':
#        from base.DynamicalSystems import DynamicalSystem
#        from base.TanhHybridEchoStateNetwork import TanhHybridEchoStateNetwork
#        
#        phys = DynamicalSystem.load(args.phys)
#        
#        esn = TanhHybridEchoStateNetwork(args.tanh_scale, phys, args.Gamma, 
#                args.N_units, args.N_dim, args.rho, args.sparseness,
#                args.sigma_in, args.activation_name, args.bias_in,
#                args.bias_out, None, None, rand, args.file_path)

    else:
        raise ValueError('Invalid ESN type {}.'.format(args.esn_type))

    esn.save()
