import os
import numpy as np
from scipy.optimize import minimize
from scipy import fftpack

def path_and_name(file_path):
    """ Returns path and name for a file_path 
        (e.g. /home/username/foo.txt returns /home/username and foo.txt)
    """
    parts = file_path.split(os.sep)
    path = os.sep.join(parts[:-1])
    name = parts[-1]
    return path, name

def error_bounds(x):
    """ Receives a series of samples x_1, ..., x_N and returns the mean and
        standard deviation of the distribution of the partial averages
        S_n = (x_1 + .. + x_n)/n based on the Central Limit Theorem by solving
        a minimization problem. The minimization problem to be solved is:
        $$
            min_{a,b} b \\
            \mathrm{subject to:}
                a - b/\sqrt(i) < S_n < a + b/\sqrt(i) \forall n \in \{1, \dots, N\}
        $$
        
        Args:
            x (np.ndarray): samples
        
        Returns:
            float: mean
            float: standard deviation
    """
    assert x.ndim == 1
    N = x.size

    S = np.cumsum(x, axis=0)/(1+np.arange(N))

    objective = lambda t: t[1]
    constraint = lambda t, n, sign: sign*((t[0]+sign*t[1]/np.sqrt(n+1))-S[n])

    constraints = []
    for i in range(N):
        # lower and upper bounds constraints on S_i
        constraints.append({'type':'ineq', 'fun':constraint, 'args':(i,-1)})
        constraints.append({'type':'ineq', 'fun':constraint, 'args':(i,1)})
    
    res = minimize(objective,
            x0=(S[-1], S.std()),
            method='SLSQP',
            constraints=constraints)

    if not res.success:
        raise Exception('Optimization not successful.')

    mu, sigma = res.x

    return mu, sigma


def fspectrum(var, dt, start=0, end=None):
    """ Frequency analysis of time series

    Args:
        case (Case): problem case
        var (array): time series of variable
        start (int): skip iterations before iteration start
        end (int): skip iterations after iteration end
    """
    if end is None:
        end = len(var)

    length = end - start

    xf = np.linspace(0.0, 1.0/(2.0*dt), length//2)
    yt = fftpack.fft(var[start:end])
    yf = 2.0/length * abs(yt[:length//2])

    return xf, yf
