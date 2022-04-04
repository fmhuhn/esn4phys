import h5py
import numpy as np
from scipy.integrate import odeint

from .Integrators import Integrator
from .Normalizations import NORMALIZATIONS
from .DynamicalSystems import DynamicalSystem

class Dataset:
    def __init__(self, u, norm, dt, integrator=None, system=None,
            file_path=None):
        """ Create Dataset intance from normalized data.

            Args:
                u (np.ndarray): dataset time series
                norm (Normalization): normalization
                dt (float): time step of time series (i.e. 1/sampling rate)
                integrator (str or function): integrator method (see
                    Integrators.INTEGRATORS for options)
                system (DynamicalSystem): dynamical system of dataset
                file_path (str): file path of dataset
        """

        self.u = u
        self.norm = norm
        self.dt = dt
        self.system = system
        self.file_path = file_path
        if isinstance(integrator, Integrator):
            self.integrator = integrator
        else:
            self.integrator = Integrator(integrator)

    @classmethod
    def by_generating_data(cls, N, norm_name, dt, integrator_name,
            system, file_path=None, rand=None, u0=None):
        """ Create Dataset instance by generating data.

            Args:
                N (int): number of time steps
                norm_name (str): name of normalization method (None if not
                dt (float): time step of time series (i.e. 1/sampling rate)
                    normalized)
                integrator_name (str): integrator method name (see
                    Integrators.INTEGRATORS for options)
                system (DynamicalSystem): dynamical system of dataset
                file_path (str): file path of dataset
                rand (mtrand.RandomState): random state to generate initial
                    condition (useful for repeatability, i.e. repeat same
                    initial condition for testing)
                u0 (array): initial condition
        """

        integrator = Integrator(integrator_name)
        if u0 is None:
            u0 = system.generate_initial_condition(rand)
        N_transient = cls.calc_N_transient(system.t_transient, dt)
        u = cls.generate_nonnorm_u(N_transient+N, dt, integrator, system, \
                 u0)[N_transient:]

        if system.name == 'rijke_with_perturbation':
            u = u[:,:-3]
            system.name = 'rijke'
            system.file_path = system.file_path.replace('.', '_m.')
            system.save()

        if norm_name:
            norm = NORMALIZATIONS[norm_name].from_dataseries(u)
            u = norm.normalize(u)
        else:
            norm = None

        return cls(u, norm, dt, integrator, system, file_path)

    @staticmethod
    def generate_nonnorm_u(N, dt, integrator, system, u0):
        """ Generates a non-normalised time series by integrating a dynamical
            system.

            Args:
                N (int): number of time steps
                dt (float): time step
                integrator (Integrator): time integrator
                system (DynamicalSystem): dynamical system
                u0 (np.ndarray): non-normalized initial condition
        """
        T = np.arange(N+1) * dt
        u = integrator.integrate(system.ddt, u0, T)
        return u

    @property
    def N(self):
        """ Number of time steps (length of dataset). """
        return self.u.shape[0]

    @property
    def N_transient(self):
        """ Number of transient time steps. """
        return self.calc_N_transient(self.system.t_transient, self.dt)

    @staticmethod
    def calc_N_transient(t_transient, dt):
        """ Number of transient time steps for given transient time and
            time step. """
        return int(np.ceil(t_transient/dt))

    def normalize(self, v):
        """ Normalize v using the normalization of the dataset.

            Args:
                v (np.ndarray): vector or timeseries (time in first axis) to 
                    be normalized
        """

        return self.norm.normalize(v)

    def denormalize(self, vn):
        """ Denormalize vn using the normalization of the dataset.

            Args:
                vn (np.ndarray): vector or timeseries (time in first axis) to 
                    be denormalized
        """
        return self.norm.denormalize(vn)

    def save(self):
        """ Save dataset file under the path in `self.file_path'. Thus, 
            `self.file_path' must have been set before calling this.
        """

        with h5py.File(self.file_path, 'w') as hf:
            hf.create_dataset('u', data=self.u)
            hf.attrs['norm_name'] = self.norm.name
            hf.attrs['norm_params'] = self.norm.params
            hf.attrs['dt'] = self.dt
            hf.attrs['integrator'] = self.integrator.name
            hf.attrs['system_fp'] = self.system.file_path

    @staticmethod
    def load(file_path):
        """ Load dataset.

            Args:
                file_path (str): path of h5 dataset file to load
        """

        with h5py.File(file_path, 'r') as hf:
            u = hf['u'][:]
            norm_name = hf.attrs['norm_name']  # .decode('utf-8')
            norm_params = hf.attrs['norm_params']
            norm = NORMALIZATIONS[norm_name](norm_params)
            dt = hf.attrs['dt']
            integrator_name = hf.attrs['integrator']  # .decode('utf-8')
            integrator = Integrator(integrator_name)
            system = DynamicalSystem.load(hf.attrs['system_fp'])

        return Dataset(u, norm, dt, integrator, system, file_path)
