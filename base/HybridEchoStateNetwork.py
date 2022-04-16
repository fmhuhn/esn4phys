import h5py
import numpy as np

from .Dataset import Dataset
from .DynamicalSystems import DynamicalSystem
from .EchoStateNetwork import EchoStateNetwork
from .Integrators import Integrator

class HybridEchoStateNetwork(EchoStateNetwork):
    def __init__(self, phys, Gamma, N_units, N_dim, rho,
            sparseness, sigma_in, activation_name, bias_in=None,
            bias_out=None, Win=None, W=None, rand=None, file_path=None):
        """ Create a Hybrid Echo State Network. If Win or W are not passed, they
            are randomly generated with the given hyperparameters.

            Args:
                phys (DynamicalSystem): physical model
                Gamma (float): fraction of nodes that receive input from
                    physical model
                *Remaining arguments are from `EchoStateNetwork.EchoStateNetwork'
        """

        self.phys = phys
        self.Gamma = Gamma

        super().__init__(N_units, N_dim+self.phys.N_dim, rho, sparseness,
            sigma_in, activation_name, bias_in, bias_out, Win, W,
            rand, file_path)

        self.N_dim = N_dim
        # replace regular ESN's Win
        self.Win = Win if Win is not None else self.build_Win()

    @property
    def xa_dim(self):
        return super().xa_dim + self.phys.N_dim

    def set_physics_numerics(self, integrator, dt, norm=None, convert_data_form=None):
        """ Set numerics for physical model.
        
            Args:
                integrator (Integrator): integrator instance
                dt (float): integration time step
                norm (Normalization): normalization instance
                convert_data_form (function): converts from ESN's input form to
                    the physics' state vector form and vice-versa
        """
        self.integrator = integrator
        self.dt = dt
        self.norm = norm
        self.convert_data_form = convert_data_form

    def advance_step(self, u, x):
        """ Advance one (discrete time) step.
        
            Args:
                u (np.ndarray): input
                x (np.ndarray): reservoir state

            Returns:
                np.ndarray: reservoir intermediate output (augmented state)
                np.ndarray: new reservoir state
        """

        # denormalize input
        if self.norm is None:
            physics_input = u
        else:
            physics_input = self.norm.denormalize(u).flatten()

        # convert input data to physical model's state vector form
        if self.convert_data_form:
            physics_input = self.convert_data_form(physics_input,
                                direction='0', other=self.phys)

        # integrate physical model ([1] -> last time step (new state))
        physics_output = self.integrator.integrate(self.phys.ddt, physics_input,
                                        [0.0, self.dt])[1]

        # normalize physics output
        if self.norm is not None:
            # convert physics output to ESN's data form
            if self.convert_data_form is not None:
                physics_output = self.convert_data_form(physics_output,
                                    direction='1', other=self.phys)
            
            physics_output = self.norm.normalize(physics_output)

            if self.convert_data_form is not None:
                physics_output = self.convert_data_form(physics_output,
                                    direction='0', other=self.phys)

        augmented_u = np.hstack([u, physics_output])
        out_xa, out_x = super().advance_step(augmented_u, x)

        return np.hstack([out_xa, physics_output]), out_x


    def build_Win(self):
        """ Constructs Win (self.N_dim+d_bias x self.N_units), where d_bias=1
            if there is an input bias and 0 otherwise. Win's entries are
            sampled from unif(-self.sigma_in, +self.sigma_in) and such that
            each only receives one input variable.
        """

        assert self.rand

        d_bias = 1 if self.bias_in else 0

        Win = np.zeros((self.N_dim+d_bias+self.phys.N_dim, self.N_units))

        n = int(np.floor(self.N_units*self.Gamma))

        # variable limits
        var_lims = [0, self.N_dim+d_bias, self.N_dim+d_bias+self.phys.N_dim]
        # node limits
        node_lims = [0, self.N_units-n, self.N_units]
        assert len(var_lims) == len(node_lims)
        # iterate over nodes, pick an input variable and generate a weight
        for i in range(len(var_lims)-1):
            lo_node, hi_node = node_lims[i], node_lims[i+1]
            lo_var, hi_var = var_lims[i], var_lims[i+1]
            for j in range(lo_node, hi_node):
                var_index = self.rand.randint(lo_var, hi_var)
                Win[var_index,j] = self.rand.uniform(-1, 1)

        return Win.T

    def save(self):
        """ Save hybrid echo state network file under the path in
            `self.file_path'.
        """

        super().save()

        with h5py.File(self.file_path, 'a') as hf:
            hf.attrs['phys_fp'] = self.phys.file_path
            hf.attrs['Gamma'] = self.Gamma

    @classmethod
    def load(cls, file_path):
        """ Load hybrid echo state network.

            Args:
                file_path (str): path of h5 hybrid echo state network file
        """
        esn = super().load(file_path)

        with h5py.File(file_path, 'r') as hf:
            phys = DynamicalSystem.load(hf.attrs['phys_fp'])
            Gamma = hf.attrs['Gamma']

        return cls(phys, Gamma, esn.N_units, esn.N_dim,
                esn.rho, esn.sparseness, esn.sigma_in, esn.activation_name,
                esn.bias_in, esn.bias_out, esn.Win, esn.W, None, file_path)
