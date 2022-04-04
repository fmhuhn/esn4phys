import h5py
import numpy as np

from .Dataset import Dataset
from .DynamicalSystems import DynamicalSystem
from .HybridEchoStateNetwork import HybridEchoStateNetwork
from .Integrators import Integrator

class TanhHybridEchoStateNetwork(HybridEchoStateNetwork):
    def __init__(self, scale, phys, Gamma, N_units, N_dim, rho,
            sparseness, sigma_in, activation_name, bias_in=None,
            bias_out=None, Win=None, W=None, rand=None, file_path=None):
        """ Create a Hybrid Echo State Network. If Win or W are not passed, they
            are randomly generated with the given hyperparameters.

            Args:
                scale (float): scaling factor for tanh of physics output (i.e.
                    tanh(scale*physics_output))
                phys (DynamicalSystem): physical model
                Gamma (float): fraction of nodes that receive input from
                    physical model
                *Remaining arguments are from `EchoStateNetwork.EchoStateNetwork'
        """

        self.scale = scale

        super().__init__(phys, Gamma, N_units, N_dim, rho,
            sparseness, sigma_in, activation_name, bias_in,
            bias_out, Win, W, rand, file_path)

    def advance_step(self, u, x):
        """ Advance one (discrete time) step.
        
            Args:
                u (np.ndarray): input
                x (np.ndarray): reservoir state

            Returns:
                np.ndarray: reservoir intermediate output (augmented state)
                np.ndarray: new reservoir state
        """

        out_xa, out_x = super().advance_step(u, x)
        out_xa[-self.phys.N_dim:] = np.tanh(self.scale*out_xa[-self.phys.N_dim:])

        return out_xa, out_x

    def save(self):
        """ Save hybrid echo state network file under the path in
            `self.file_path'.
        """

        super().save()

        with h5py.File(self.file_path, 'a') as hf:
            hf.attrs['scale'] = self.scale

    @classmethod
    def load(cls, file_path):
        """ Load hybrid echo state network.

            Args:
                file_path (str): path of h5 hybrid echo state network file
        """

        hesn = HybridEchoStateNetwork.load(file_path)

        with h5py.File(file_path, 'r') as hf:
            scale = hf.attrs['scale']

        return cls(scale, hesn.phys, hesn.Gamma, hesn.N_units, hesn.N_dim,
                hesn.rho, hesn.sparseness, hesn.sigma_in, hesn.activation_name,
                hesn.bias_in, hesn.bias_out, hesn.Win, hesn.W, None, file_path)
