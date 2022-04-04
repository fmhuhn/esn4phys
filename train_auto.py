import os
import argparse
import warnings

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import skopt

from base.Case import Case
import base.Train as Train


def bayesian_hyperparameter_search(case): 
    global rand
    rand = np.random.RandomState(0)

    rho_min, rho_max = 1e-3, 1e0
    sigma_min, sigma_max = 1e-2, 1e2
    # tikh_min, tikh_max = 1e-11, 1e-5

    max_cycles = 10
    n_calls_per_cycle = 5
    n_random_starts = 5

    threshold = np.log10(1e-2)
    obj = 3*threshold
    x0 = None
    y0 = None
    
    global hcount
    hcount = 0

    for i_cycle in range(max_cycles):
        res = skopt.gp_minimize(
                train_and_validate,
                [(rho_min, rho_max, 'log-uniform'),
                (sigma_min, sigma_max, 'log-uniform')],
                x0=x0,
                y0=y0,
                acq_func='gp_hedge',
                n_calls=n_calls_per_cycle,
                n_initial_points=n_random_starts,
                initial_point_generator='lhs',
                noise=0.2**2,
                random_state=3
            )

        x0 = res.x_iters
        y0 = res.func_vals
        obj = res.fun

        if obj<threshold:
            break

        n_calls_per_cycle = 5
        n_random_starts = 0

    case.esn.rho, case.esn.sigma_in = res.x
    train_and_validate(res.x, plot=False)

    return case, res

def train_and_validate(hyperparams, plot=False):
    global case
    case.esn.rho, case.esn.sigma_in = hyperparams

    Wout, Xa_train = Train.train_network(case.esn, ts_skip, ts_train[:-1], ts_train[1:],
            xa0=np.zeros(case.esn.xa_dim), tikh=case.tikh)
    case.Wout = Wout

    Yh_valid, _ = case.esn.closed_loop(N_valid, Xa_train[-1], Wout)
    MSE = np.mean((Yh_valid-ts_valid)**2)

    if MSE > MSE_saturation or np.isnan(MSE):
        warnings.warn('MSE {0:e} saturated to {1:e}'.format(MSE, MSE_saturation))
        MSE = MSE_saturation
    
    if plot:
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        ax[0].plot(ts_valid[:,0], ts_valid[:,10])
        ax[0].plot(Yh_valid[:,0], Yh_valid[:,10], linestyle='--', alpha=0.5)
        
        T = np.arange(N_valid+1)*case.ds.dt
        u, _, _ = case.ds.system.fourierToReal(ts_valid, x=0.2)
        uh, _, _ = case.ds.system.fourierToReal(Yh_valid, x=0.2)
        ax[1].plot(T, u)
        ax[1].plot(T, uh, linestyle='--', alpha=0.5)
        plt.show()

    global count
    count += 1
    print("%d %.2e %.2e %.2e %e" % (count, case.esn.rho, case.esn.sigma_in, case.tikh, MSE))

    return np.log10(MSE)


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
        case.esn.set_physics_numerics(case.ds.integrator, case.ds.dt, case.ds.norm, case.ds.system.convert_data_form)

    case.tikh = 1e-9
    case.N_skip = 500

    assert args.N_valid or args.F_valid
    N_valid = args.N_valid if args.N_valid else int(args.F_valid*case.N_train)
    N_test = N_valid

    split_lengths = [case.N_skip, case.N_train, N_valid, N_test]
    overlap = [0, 1, 1, 0]
    ts_skip, ts_train, ts_valid, ts_test = Train.split_timeseries(case.ds.u, split_lengths, overlap)

    count = 0
    rand = np.random.RandomState(0)
    MSE_saturation = 1e3

    case, res = bayesian_hyperparameter_search(case)
    case.save()
