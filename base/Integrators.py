import numpy as np
from scipy.integrate import odeint

def forward_euler(ddt, u0, T, *args):
    u = np.empty((len(T), len(u0)))
    u[0] = u0
    for i in range(1, len(T)):
        u[i] = u[i-1] + (T[i] - T[i-1]) * ddt(u[i-1], T[i-1], *args)
    return u


ALPHA, GAMMA, ETA = [0,0,0], [8./15, 5./12, 3./4], [0, -17./60, -5./12]
def wray_rk(ddt, q0, T, args=()):
    q = np.empty((len(T), 3, len(q0)))
    Q = q0.copy()
    N0, N1 = np.empty((2, q0.size))

    for i in range(1, len(T)):
        dt = T[i] - T[i-1]
        for k in range(3):
            q[i-1, k, :] = Q
            N0 = ddt(Q, T[i-1] + ALPHA[k] + dt, *args)
            Q += dt * (GAMMA[k] * N0 + ETA[k] * N1)
            N1[:] = N0
    q[-1, 0] = Q

    return q[:,0,:]


class Integrator:
    integrators = {
                    'forward_euler': forward_euler,
                    'odeint': odeint,
                    'wray_rk': wray_rk
                  }

    def __init__(self, integrator_name):
        self.name = integrator_name
        self.integrate = self.integrators[integrator_name]
