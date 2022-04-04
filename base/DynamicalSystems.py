import json

import numpy as np


class DynamicalSystem:
    # to be defined by children
    file_path = None
    param_names = []
    param_vals = []
    name = None

    def __init__(self, params=None, file_path=None):
        """ Create a dynamical system instance.

            Args:
                param_vals (dict or np.ndarray): model parameters.
                    dict - keys must match `DynamicalSystem.param_names'.
                    np.ndarray - must be in the order of
                        `DynamicalSystem.param_names'.
                    None - default parameters are used.
                file_path (str): file path of dynamical system file.
        """
        if isinstance(params, dict):
            param_vals = []
            for p_name in self.param_names:
                setattr(self, p_name, params[p_name])
                param_vals.append(params[p_name])
            self.param_vals = tuple(param_vals)

        else:
            if params is not None:
                self.param_vals = params

            for p_name, p_val in zip(self.param_names, self.param_vals):
                setattr(self, p_name, p_val)

        self.file_path = file_path

    def save(self):
        """ Save dynamical system file under the path in `self.file_path'.
            Thus, `self.file_path' must have been set before calling this.
        """

        with open(self.file_path, 'w') as case_file:
            # data = dict(zip(self.param_names, self.param_vals))
            data = dict()
            data['name'] = self.name
            for p_name, p_val in zip(self.param_names, self.param_vals):
                data[p_name] = p_val
            json.dump(data, case_file, sort_keys=False, indent=2)

    @classmethod
    def load(cls, file_path):
        """ Load dynamical system file.

            Args:
                file_path (str): path of json case file to load
        """
        
        with open(file_path) as case_file:
            data = json.load(case_file)
            name = data.pop('name')
            if cls.name:
                assert name == cls.name
            System = cls.get_from_name(name)
            return System(data, file_path)

    @staticmethod
    def get_from_name(name):
        return all_systems[name]


class lorenz63(DynamicalSystem):
    N_dim = 3
    param_names = ['beta', 'rho', 'sigma']
    param_vals = (8/3., 28.0, 10.0)  # default param values
    name = 'lorenz63'
    t_transient = 10.0
    lyap_time = 0.9**-1

    def ddt(self, q, t):
        """ Right-hand side, F, of the usual dynamical system equation
            dq/dt = F(q).

            Args:
                q (np.ndarray): state vector
                t (float): time (irrelevant in autonomous systems)
            
            Returns:
                np.ndarray: time-derivative of state vector
        """
        x, y, z = q
        return np.array([self.sigma*(y-x), x*(self.rho-z)-y, x*y-self.beta*z])

    @staticmethod
    def generate_initial_condition(rand=None):
        """ Generates a random initial condition that should lead to the 
            attractor in lorenz63.t_transient time, although it's a guess.

            Args:
                rand (mtrand.RandomState): random state (useful for
                    repeatability, i.e. repeat same initial condition
                    for testing)
        """

        if rand is None:
            rand = np.random.RandomState()
        u0 = np.zeros(lorenz63.N_dim)
        u0[0] = 25.0
        u0[1:] = rand.rand(lorenz63.N_dim-1)
        return u0

    def convert_data_form(self, u, direction, other):
        return u

class Rijke(DynamicalSystem):
    param_names = ['beta', 'tau', 'Ng', 'Nc', 'xf', 'c1', 'c2']
    param_vals = (7.0, 0.2, 10, 10, 0.2, 0.1, 0.06)  # default param values
    name = 'rijke'
    t_transient = 200.0
    lyap_time = 0.12**-1

    @property
    def N_dim(self):
        return 2*self.Ng + self.Nc

    @property
    def j(self):
        if not hasattr(self, '_j'):
            self._j = 1+np.arange(self.Ng)
        return self._j

    @property
    def jpi(self):
        if not hasattr(self, '_jpi'):
            self._jpi = self.j * np.pi
        return self._jpi

    @property
    def damping(self):
        if not hasattr(self, '_damping'):
            self._damping = self.c1*self.j**2 + self.c2*self.j**0.5
        return self._damping

    @property
    def cosjpixf(self):
        if not hasattr(self, '_cosjpixf'):
            self._cosjpixf = np.cos(self.jpi * self.xf)
        return self._cosjpixf

    @property
    def sinjpixf(self):
        if not hasattr(self, '_sinjpixf'):
            self._sinjpixf = np.sin(self.jpi * self.xf)
        return self._sinjpixf

    def split(self, q):
        if q.ndim == 1:
            assert q.size == self.N_dim
            eta, mu, v = q[:self.Ng], q[self.Ng:2*self.Ng], q[2*self.Ng:]
        else:
            assert q.shape[1] == self.N_dim
            eta, mu, v = q[:,:self.Ng], q[:,self.Ng:2*self.Ng], q[:,2*self.Ng:]
        return eta, mu, v

    @property
    def chebgrid(self):
        if not hasattr(self, '_chebgrid'):
            n = self.Nc
            self._chebgrid = -np.cos(np.pi*np.arange(n+1, dtype=float)/n)
        return self._chebgrid

    @property
    def chebdiff(self):
        if not hasattr(self, '_chebdiff'):
            x = self.chebgrid
            n = self.Nc
            c = np.hstack([2., np.ones(n-1), 2.]) * (-1)**np.arange(n+1)
            X = np.outer(x, np.ones(n+1))
            dX = X-X.T
            self._chebdiff = np.outer(c, 1/c)/(dX + np.eye(n+1))
            self._chebdiff -= np.diag(self._chebdiff.sum(1))

        return self._chebdiff

    def q_dot(self, delayed_velocity):
        # if abs(delayed_velocity + 1.0) < 0.01:
        #     coeffs = np.array([-1.0, 0.0, 1.75e3, 6.2e-12, -7.5e6])
        #     poly = np.poly1d(coeffs[::-1])
        #     return poly(1.0 + delayed_velocity)

        return np.sqrt(np.abs(1.0 + delayed_velocity)) - 1.0

    def ddt(self, q, t):
        eta, mu, velocity = self.split(q)

        velocity_at_flame = np.dot(eta, self.cosjpixf)
        velocity = np.hstack([velocity_at_flame, velocity])
        heat_release = self.beta * self.q_dot(velocity[-1])

        ddt_eta = self.jpi * mu
        ddt_mu = -(self.jpi*eta+self.damping*mu+2*self.sinjpixf*heat_release)
        ddt_velocity = -2 * np.dot(self.chebdiff, velocity)[1:] / self.tau

        return np.hstack([ddt_eta, ddt_mu, ddt_velocity])

    def fourierToReal(self, q=None, x=None, nx=None):
        if x is None:
            if nx is None:
                raise ValueError("Both x and nx can't be None")
            x = np.linspace(0.0, 1.0, nx)

        x = np.array(x)
        jpix = self.jpi.reshape((self.Ng, 1)) * x

        cosjpix = np.cos(jpix)
        sinjpix = np.sin(jpix)

        eta, mu, v = self.split(q)

        return np.dot(eta, cosjpix), -np.dot(mu, sinjpix), v

    def generate_initial_condition(self, rand=None):
        """ Generates a random initial condition that should lead to the 
            attractor in Rijke.t_transient time, although it's a guess.

            Args:
                rand (mtrand.RandomState): random state (useful for
                    repeatability, i.e. repeat same initial condition
                    for testing)
        """

        if rand is None:
            rand = np.random.RandomState()
        u0 = np.zeros(self.N_dim)
        u0[0] = 1.0
        u0[1:] = rand.rand(self.N_dim-1)
        return u0

    def compute_Eac(self, u):
        eta, mu, _ = self.split(u)
        return 0.25*(eta**2 + mu**2).sum(axis=1)

    def convert_data_form(self, u, direction, other):
        assert self.Nc == other.Nc
        assert self.Ng >= other.Ng

        if direction == '0':  # from self to other
            eta, mu, v = self.split(u)
            eta = eta[:other.Ng]
            mu = mu[:other.Ng]
            
        elif direction == '1': # from other to self
            eta, mu, v = other.split(u)
            eta = np.pad(eta, pad_width=(0, self.Ng-other.Ng),
                    constant_values=0)
            mu = np.pad(mu, pad_width=(0, self.Ng-other.Ng),
                    constant_values=0)
        output = np.hstack([eta, mu, v])
        return output


all_systems = {'lorenz63': lorenz63, 'rijke': Rijke}
