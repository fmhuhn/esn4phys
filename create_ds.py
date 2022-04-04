import os
import sys
import argparse

import numpy as np

from base.Dataset import Dataset
from base.DynamicalSystems import DynamicalSystem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset.')

    parser.add_argument('N', metavar='N', type=int,
        help='length of dataset')
    
    parser.add_argument('norm_name', metavar='norm_name', type=str,
        help='normalization method')
    
    parser.add_argument('dt', metavar='dt', type=float,
        help='time step of dataset')
    
    parser.add_argument('integrator_name', metavar='integ_name', type=str,
        help='number of time steps for trainng')
    
    parser.add_argument('system_fp', metavar='system_fp', type=str,
        help='dynamical system file path')
    
    parser.add_argument('--rand_seed', metavar='rand_seed', type=int,
        help='initial condition random seed')
    
    parser.add_argument('file_path', metavar='file_path', type=str,
        help='file path of case file')

    args = parser.parse_args()

    system = DynamicalSystem.load(args.system_fp)
    dataset = Dataset.by_generating_data(args.N, args.norm_name, args.dt,
                args.integrator_name, system, args.file_path)
    dataset.save()
