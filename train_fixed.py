import os
import sys
import argparse

import numpy as np
from base.Case import Case
import base.Train as Train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an ESN case.')

    parser.add_argument('case', metavar='case_file_path', type=str,
            help='case file')

    parser.add_argument('N_valid', metavar='N_valid', type=int,
            help='number of time steps for validation')


    parser.add_argument('--esn_type', metavar='esn_type', type=str,
        help='type of echo state network (normal, hybrid, tanh_hybrid',
        default='normal')

    args = parser.parse_args()

    case = Case.load(args.case, args.esn_type)

    if args.esn_type == 'hybrid':
        case.esn.set_physics_numerics(case.ds.integrator, case.ds.dt, case.ds.norm,
                            case.ds.system.convert_data_form)

    esn = case.esn
    ds = case.ds

    split_lengths = [case.N_skip, case.N_train, args.N_valid]
    overlap = [0, 1, 1]
    ts_skip, ts_train, ts_valid = Train.split_timeseries(case.ds.u, split_lengths, overlap)

    Wout, Xa_train = Train.train_network(esn, ts_skip, ts_train[:-1], ts_train[1:], xa0=np.zeros(case.esn.xa_dim), tikh=case.tikh)
    Yh_valid, _ = case.esn.closed_loop(args.N_valid, Xa_train[-1], Wout)
    MSE = np.mean((Yh_valid-ts_valid)**2)
    
    print('MSE:', MSE)

    case.Wout = Wout
    case.save()
