import numpy as np


class VanDerPol:

    def __init__(self, mu, dt):
        self.mu = mu  # system parameter
        self.dt = dt  # time step
        self.dt2 = dt * dt  # dt²

    def next_output(self, u, y0, y1):
        return 2 * y1 - y0 + self.mu * self.dt * (1 - y0 * y0) * (y1 - y0) + self.dt2 * (u - y0)

    def run(self, u, t0):
        ts = t0 + np.array([self.dt * i for i in range(len(u))])
        y = np.zeros(len(ts))

        # initial conditions:
        y[0] = 0.0
        dy_dt = 0.0
        y[1] = y[0] + self.dt * dy_dt

        for i in range(2, len(ts)):
            y[i] = self.next_output(u[i], y[i - 2], y[i - 1])

        return ts, y

    def run_controlled(self, reference, t0, controller):
        ts = t0 + np.array([self.dt * i for i in range(len(reference))])
        y = np.zeros(len(ts))
        u = np.zeros(len(ts))

        # initial conditions:
        y[0] = 0.0
        dy_dt = 0.0
        y[1] = y[0] + self.dt * dy_dt

        for i in range(2, len(ts)):
            u[i] = controller.compute(reference[i] - y[i - 1], self.dt)
            y[i] = self.next_output(u[i], y[i - 2], y[i - 1])

        return ts, y, u
