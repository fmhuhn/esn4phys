# esn4phys

**esn4phys** is an implementation of leakless echo state networks and variants, such as the hybrid echo state network, for the prediction of physical systems, usually chaotic.

Echo state networks (ESN) [1] are recurrent neural networks (RNN) where only the output weights are trained. This makes training easy and cheap, as the training problem reduces to ridge regression. Moreover, the computational cost is also reduced because ESNs are usually sparse. A practical guide on ESNs can be found in [2].

This project includes implementations of the (leakless) conventional echo state network and the hybrid echo state network (hESN) [3]. In the hESN, the reservoir of the ESN is complemented wth physical knowledge in the form of a dynamical system (e.g. a reduced-order model), which usually results in increased performance [4]. Additionally, it also includes the tanh variant of the hESN [4].

The ESN code is structured in a way that variants can be easily built upon it. The project also includes automatic hyperparameter optimisation with Bayesian optimisation [4,5].

## Structure

The code of this project is split into base and scripts. The base folder contains all the "core" code of the project (e.g. `Case`, `Dataset`, `DynamicalSystem`, `EchoStateNetwork`, etc.). The scripts are in the root directory of the project and they are `train_XXXXX.py` (e.g `train_fixed.py`), which train the network, and `create_XXXXX.py` scripts (e.g. `create_case.py`), which create the necessary datasets, ESNs, dynamical systems, etc. files.

The folder `tests/` contains simple examples of predictions. It has four directories, each one to store cases, datasets, dynamical systems and network files. It's not mandatory, as the files can be located anywhere. Just make sure not to move them/rename them, because the links between files will be broken.

## Dynamical System

Starting from scratch (i.e. implementing a new dynamical system), one must code the dynamical system in `base/DynamicalSystems.py`. To implement a dynamical system, a class that inherits from `DynamicalSystem` must be created with the methods `generate_initial_condition`, which generates an initial condition, and `ddt`, which computes the time derivative, must be created too. It must also feature the class attributes `param_names`, `param_vals` and `name`, which contain the names of the physical parameters of the system, the default values of those parameters and the name of the system. Finally, at the end of the file, the class must be added to the dictionary `all_systems`, with the key being equal to the name attribute and the value being the class name. See the `lorenz63` class for a simple example.

The previous instructions implement the dynamical system, but a dynamical system file must be created to use it. This can be done with the `create_sys.py` script. To create the file run:

```
$ python3 create_sys.py system_name file_path --params param_vals
```

For example, for the Lorenz system with `beta=2.0`, `rho=28.0`, `sigma=10.0`; and store it in the `dynsystems/` folder (in `tests/`) with the name `lorenz63_beta_2p0_rho_28p0_sigma_10p0.sys`, run:

```
$ python3 create_sys.py lorenz63 tests/dynsystems/lorenz63_beta_2p0_rho_28p0_sigma_10p0.sys --params 2.0 28.0 10.0 
```

The created file is a json file and the extension `.sys` is arbitrary.


## Dataset

With the dynamical system implemented and the system file created, you can create a dataset, i.e. the numerical solution obtained from time-integrating the dynamical system. To create the dataset, run:

```
$ python3 create_ds.py N norm_name dt integ_name system_fp file_path [--rand_seed rand_seed]
```

Arguments:

- `N`            number of time steps
- `norm_name`    name of the data normalization (see ### Normalisation)
- `dt`           time step
- `integ_name`   name of the integrator (see ### Integrators)
- `system_fp`    filepath of the dynamical system file (e.g. in the example of the previous section, it would be dynsystems/lorenz63_beta_2p0_rho_28p0_sigma_10p0.sys)
- `file_path`    filepath of the dataset file being created

The optional argument `--rand_seed rand_seed` is the seed to generate the random initial condition.

For example, to create a dataset with `1000` time steps, `max_minus_min` normalization, a `0.01` time step, `wray_rk` integrator, using the dynamical system file `dynsystems/lorenz63_beta_2p0_rho_28p0_sigma_10p0.sys`, and with an initial condition random seed `0`; to be stored in `datasets/` with the name `lorenz63_beta_2p0_rho_28p0_sigma_10p0.ds`, run:

```
$ python3 create_ds.py 1000 max_minus_min 0.01 wray_rk tests/dynsystems/lorenz63_beta_2p0_rho_28p0_sigma_10p0.sys tests/datasets/lorenz63_beta_2p0_rho_28p0_sigma_10p0.ds --rand_seed 0
```

The created file is an h5 file and the extension `.ds` is arbitrary.

### Normalizations

The normalization of the data is important to get good results. These are implemented in `base/Normalizations.py`. Currently only `max_minus_min` is implemented, which divides each state variable by the difference between its maximum minus minimum.

### Integrators

The time integrators are implemented in `base/Integrators.py`. Currently, there are three: forward Euler, odeint and Wray's Runge-Kutta. odeint is not recommended due to its adaptive time step. Implementation of new integrators is straightforward.


## Echo State Network

To create an ESN, one must run the `create_esn.py` script:

```
$ python3 create_esn.py N_units N_dim rho sigma_in file_path
```

Arguments:

- `N_units`     size of the reservoir
- `N_dim`       size of the input/output of the network
- `rho`         spectral radius of W
- `sigma_in`    input scaling parameter
- `file_path`   filepath of ESN file being created

You can also include the following optional arguments:

- `--sparseness sp`         sp is the reservoir sparseness, between 0 and 1. If omitted, the sparseness will be `1-3/(N_units-1)`
- `--bias_in b_in`          b_in is the input bias
- `--bias_out b_out`        b_out is the output bias
- `--rand_seed rand_seed`   rand_seed is the random seed to generate the "random" reservoir
- `--phys system`           see "Hybrid Echo State Network"
- `--Gamma Gamma`           see "Hybrid Echo State Network"
- `--esn_type esn_type`     `esn_type` is `normal` for conventional ESN or `hybrid` for hybrid ESN (see "Hybrid Echo State Network")

For example, to create a network with 100 nodes, 3-dimensional output (e.g. to predict the Lorenz system), `rho=0.3` and `sigma_in=1.0`, an output bias of `1.0` and the random seed `0`, to be stored in `networks/` with the name `Nx_100_Nd_3.esn`, run:

```
$ python3 create_esn.py 100 3 0.3 1.0 tests/networks/Nx_100_Nd_3.esn --bias_out 1.0 --rand_seed 0
```

### Hybrid Echo State Network

To create an hESN, which uses knowledge from a dynamical system (e.g. a ROM), one must use the `--phys`, `--Gamma` and `--esn_type` optional arguments.

For example, for the Lorenz system, we'll use a system where the physical parameter beta is slightly wrong (e.g. measurement error) at 2.1 instead of 2.0. To do that, we must first create the dynamical system file with `beta=2.1`.

```
$ python3 create_sys.py lorenz63 tests/dynsystems/lorenz63_beta_2p1_rho_28p0_sigma_10p0.sys --params 2.1 28.0 10.0 
```

Now, we can create the hESN:

```
$ python3 create_esn.py 100 3 0.3 1.0 tests/networks/Nx_100_Nd_3_K_beta_2p1.hesn --rand_seed 0 --phys tests/dynsystems/lorenz63_beta_2p1_rho_28p0_sigma_10p0.sys --Gamma 0.2 --esn_type hybrid
```

Notice the omission of the output bias. In many cases, this can be done because the knowledge model already provides an implicit bias. The argument `--Gamma Gamma` is the the fraction of reservoir nodes that receive their input from the knowledge model instead of the network's input.

In this case, the knowledge model has the same dimension as the dynamical system that generates the data and which is to be predicted. That is not necessary. For example, in the Rijke system, the knowledge model is usually one with the same parameter values as the data, but with a lower number of Galerkin modes (see the two `.sys` files in dynsystems/).


## Cases

A case is a combination of dynamical system, dataset and network. To create it, run:

```
$ python3 create_case.py esn_file_path ds_file_path N_train N_skip tikh file_path
```

Arguments:

- `esn_file_path`       filepath of the network
- `ds_file_path`        filepath of the dataset
- `N_train`             number of training samples
- `N_skip`              number of ESN transient steps
- `tikh`                Tikhonov factor
- `file_path`           filepath of case file being created

Note that `N_train+N_skip+1` must be no larger than the number of samples in the dataset. Ideally, the dataset should be long enough to cover training, skip, validation and test phases, though only the first two are mandatory.

Continuing the example, we'll create two cases, one for the conventional ESN and one for the hESN:

```
$ python3 create_case.py tests/networks/Nx_100_Nd_3.esn tests/datasets/lorenz63_beta_2p0_rho_28p0_sigma_10p0.ds 1000 100 1e-9 tests/cases/lorenz.case
$ python3 create_case.py tests/networks/Nx_100_Nd_3_K_beta_2p1.hesn tests/datasets/lorenz63_beta_2p0_rho_28p0_sigma_10p0.ds 1000 100 1e-9 tests/cases/lorenz_hybrid.case
```

## Training

There are two ways to train the network:

- `train_fixed.py`: fixed hyperparameters (`rho` and `sigma_in`), i.e. use the hyperparameter values set when the network file was created.
- `train_auto.py`: automatic hyperparameter tuning, i.e. `rho` and `sigma_in` are varied and automatically selected, using Bayesian optimisation.

To do this, run:

```
$ python3 script.py case_file_path N_valid [--esn_type esn_type]
```

where `script.py` is `train_fixed.py` or `train_auto.py`, depending on what method is to be used.

Arguments:

- `case_file_path`      filepath of the case
- `N_valid`             number of validation steps (to calculate validation MSE)
- `--esn_type`          normal or hybrid (optional)

For example:

```
$ python3 train_fixed.py tests/cases/lorenz.case 500
$ python3 bayesian_train.py tests/cases/lorenz_hybrid.case 500 --esn_type hybrid
```

## Plotting or generating results

Jupyter notebooks are a good tool for running, plotting and analysing. For example, in `lorenz_predictions.ipynb`, the predictions of the two example cases (`cases/lorenz.case` and `cases/lorenz_hybrid.case`) are compared against the test set (what's left over in the dataset after skip, training and validation).

## References

- [1] Jaeger, H. http://www.scholarpedia.org/w/index.php?title=Echo_state_network

- [2] Lukoševičius, M. 2012 A Practical Guide to Applying Echo State Networks, pp. 659–686. Springer Berlin Heidelberg.

- [3] Pathak, J. et al. 2018 Hybrid forecasting of chaotic processes: Using machine learning in conjunction with a knowledge-based model. Chaos: An Interdisciplinary Journal of Nonlinear Science 28 (4), 041101.

- [4] Huhn, F. and Magri, L. 2022 Gradient-free optimization of chaotic acoustics with reservoir computing. Phys. Rev. Fluids 7, 014402.

- [5] Racca, A. & Magri, L. 2021 Robust Optimization and Validation of Echo State Networks for learning chaotic dynamics. Neural Networks 142, 252–268.
