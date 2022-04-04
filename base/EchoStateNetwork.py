import warnings

import os
os.environ['OMP_NUM_THREADS'] = '1'

import h5py
import numpy as np

class EchoStateNetwork:
    ACTIVATIONS = {'tanh': np.tanh}

    def __init__(self, N_units, N_dim, rho, sparseness, sigma_in, activation_name,
                 bias_in=None, bias_out=None, Win=None, W=None, rand=None,
                 file_path=None):
        """ Create an Echo State Network. If Win or W are not passed, they
            are randomly generated with the given hyperparameters.

            Args:
                N_units (int): number of units in the ESN cell
                N_dim (int): dimension of input/output
                rho (float): target spectral radius
                sparseness (float): fraction of sparseness of W (between 0 and 1)
                sigma_in (float): range of Win entries, i.e. Win's entries are
                    randomly generated from unif(-sigma_in, +sigma_in)
                activation_name (str): name of nonlinearity (options in
                    EchoStateNetwork.ACTIVATIONS)
                bias_in (float): input bias
                bias_out (float): output bias
                Win (np.ndarray): input matrix (N_dim x N_units)
                W (np.ndarray): recurrent matrix (N_units x N_units)
                rand (np.random.RandomState): random number generator
                file_path (str): file path of ESN h5 file
        """

        self.N_units = N_units
        self.N_dim = N_dim
        self.rho = rho
        self.sparseness = sparseness if sparseness else 1-3/(self.N_units-1)
        self.sigma_in = sigma_in
        self.activation_name = activation_name
        self.bias_in = bias_in
        self.bias_out = bias_out
        self.rand = rand
        self.file_path = file_path

        self.Win = self.build_Win() if Win is None else Win
        # TODO: implement sparse matrix for W
        self.W = self.build_W() if W is None else W

        self.activation = EchoStateNetwork.ACTIVATIONS[activation_name]

    def advance_step(self, u, x, save=False):
        """ Advance one (discrete time) step.
        
            Args:
                u (np.ndarray): input
                x (np.ndarray): reservoir state

            Returns:
                np.ndarray: reservoir intermediate output (augmented state)
                np.ndarray: new reservoir state
        """

        res_in = np.hstack((self.bias_in, u)) if self.bias_in else u
        new_x = self.activation(np.dot(res_in, self.sigma_in*self.Win) + np.dot(x, self.rho*self.W))
            
        new_xa = np.hstack((new_x, self.bias_out)) if self.bias_out else new_x

        return new_xa, new_x

    @property
    def xa_dim(self):
        d_bias = 1 if self.bias_out else 0
        return self.N_units+d_bias

    def build_Win(self):
        """ Constructs Win (self.N_dim+d_bias x self.N_units), where d_bias=1
            if there is an input bias and 0 otherwise. Win's entries are
            sampled from unif(-self.sigma_in, +self.sigma_in) and such that
            each only receives one input variable.
        """

        assert self.rand

        d_bias = 1 if self.bias_in else 0

        Win = np.zeros((self.N_dim+d_bias, self.N_units))

        # iterate over nodes, pick an input variable and generate a weight
        for i in range(self.N_units):
            var_index = self.rand.randint(0, self.N_dim+d_bias)
            Win[var_index,i] = self.rand.uniform(-1, 1)

        return Win

    def build_W(self):
        """ Constructs recurrent matrix W (N_units x N_units) randomly, where
            each entry is taken from unif(-1, +1). Then, all but a fraction, 
            given by `self.sparseness', of the entries are set to 0 (i.e. if
            sparseness is 0.9, 90% of the entries will be 0). W is finally re-
            scaled such that its spectral radius is `self.rho'.
        """

        assert self.rand

        W_dense = self.rand.uniform(-1.0, 1.0, (self.N_units, self.N_units))

        # The filter_matrix is composed of floats between 0 and 1, which are then 
        # rounded to 0 or 1 depending if they are below or above the 1-sparseness.
        # The higher the sparseness, the larger the number of 0 entries.
        filter_matrix = self.rand.rand(self.N_units, self.N_units) < (1-self.sparseness)
        W_sparse = W_dense * filter_matrix

        # Re-scale W's spectral radius to self.rho
        eigvals = np.abs(np.linalg.eigvals(W_sparse))
        W = W_sparse / eigvals.max()

        return W

    def save(self):
        """ Save echo state network file under the path in `self.file_path'.
            Thus, `self.file_path' must have been set before calling this.
        """

#        if os.path.exists(self.file_path):
#            raise Exception(self.file_path + ' already exists.')

        with h5py.File(self.file_path, 'w') as hf:
            hf.attrs['N_units'] = self.N_units
            hf.attrs['N_dim'] = self.N_dim
            hf.attrs['rho'] = self.rho
            hf.attrs['sparseness'] = self.sparseness
            hf.attrs['sigma_in'] = self.sigma_in
            hf.attrs['activation_name'] = np.string_(self.activation_name)
            if self.bias_in:
                hf.attrs['bias_in'] = self.bias_in
            if self.bias_out:
                hf.attrs['bias_out'] = self.bias_out

            hf.create_dataset('Win', data=self.Win)
            hf.create_dataset('W', data=self.W)

            # TODO: save rand state

    @staticmethod
    def load(file_path):
        """ Load echo state network.

            Args:
                file_path (str): path of h5 echo state network file to load
        """

        with h5py.File(file_path, 'r') as hf:
            N_units = hf.attrs['N_units']
            N_dim = hf.attrs['N_dim']
            rho = hf.attrs['rho']
            sparseness = hf.attrs['sparseness']
            sigma_in = hf.attrs['sigma_in']
            activation_name = hf.attrs['activation_name'].astype('U')
            bias_in = hf.attrs['bias_in'] if 'bias_in' in hf.attrs.keys() else None
            bias_out = hf.attrs['bias_out'] if 'bias_out' in hf.attrs.keys() else None

            Win = hf['Win'][:]
            W = hf['W'][:]

        return EchoStateNetwork(N_units, N_dim, rho, sparseness, sigma_in,
                activation_name, bias_in, bias_out, Win, W, None, file_path)

    def open_loop(self, U, xa0):
        """ Advances ESN in open-loop.
            Args:
                U: input time series
                xa0: initial (augmented) reservoir state
            Returns:
                time series of augmented reservoir states
                final (unaugmented) reservoir state
        """
        N = U.shape[0]
        Xa = np.empty((N+1, self.xa_dim))
        Xa[0] = xa0
        for i in 1+np.arange(N):
            Xa[i], x = self.advance_step(U[i-1], Xa[i-1,:self.N_units])

        return Xa, x

    def closed_loop(self, N, xa0, Wout, Yh0=None):
        """ Advances ESN in closed-loop.
            Args:
                N: number of time steps
                xa0: initial augmented reservoir state
                Wout: output matrix
            Returns:
                time series of prediction
                final augmented reservoir state
        """
        xa = xa0
        Yh = np.empty((N+1, self.N_dim))
        if Yh0 is None:
            Yh[0] = np.dot(xa, Wout)
        else:
            Yh[0] = Yh0

        try:
            for i in 1+np.arange(N):
                xa, _ = self.advance_step(Yh[i-1], xa[:self.N_units])
                Yh[i] = np.dot(xa, Wout)
        except OverflowError:
            warnings.warn('Overflow during closed loop at iteration {}.'.format(i), RuntimeWarning)
            Yh[i:] = np.inf

        return Yh, xa
