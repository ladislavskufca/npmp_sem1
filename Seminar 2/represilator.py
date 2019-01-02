import yaml
import math
import numpy as np
from random import randint
import matplotlib.pylab as plt
import numpy.matlib
import time
from scipy.integrate import ode
from scipy.signal import find_peaks

import npmp_utils as my_utils

# Set random seed
RANDOM_SEED = 1234
START_TIME = 1000


class Repressilator_S_PDE:
    def __init__(self):
        # ali rob predstavlja konec prostora ali so meje neskoncne?
        # Brez sinhronizacijske molekule: periodic_bounds = 1
        # Z difuzijo sinhronizacijske molekule: periodic_bounds = 0
        self.periodic_bounds = 1

        # nalaganje shranjene konfiguracije?
        self.load_conf = 0

        # shranjevanje koncne konfiguracije?
        self.save_conf = 0

        # fiksni robovi ali spremenljivi
        self.borderfixed = 0

        # params - DEFAULT values
        self.Kd = 10
        self.alpha = 5
        self.alpha0 = 0.001 * self.alpha
        self.delta_m = 0.1
        self.delta_p = 0.1
        self.n = 2
        self.beta = 1
        self.kappa = 0.2
        self.kS0 = 1
        self.kS1 = 0.01
        self.kSe = 0.01
        self.D1 = 0.5
        self.eta = 2
        self.size = 10
        self.density = 0.4
        self.n_cells = 0
        self.dt = 0.1
        self.h2 = 0.25
        self.S_e = 0
        self.S_i = 0
        # time parameters
        self.t_end = 10000
        self.t_start = START_TIME

    def load_params(self):
        # Read YAML file
        with open("params.yaml", 'r') as stream:
            p = yaml.load(stream)

        # nalaganje vrednosti parametrov
        # p = load('params.mat')
        self.alpha = p['alpha']
        self.alpha0 = 0.001 * self.alpha
        self.Kd = p['Kd']
        self.delta_m = p['delta_m']
        self.delta_p = p['delta_p']
        self.n = p['n']
        self.beta = p['beta']
        self.kappa = p['kappa']
        self.kS0 = p['kS0']
        self.kS1 = p['kS1']
        self.kSe = p['kSe']
        self.D1 = p['D1']
        self.eta = p['eta']

        self.size = p['size']
        self.density = p['density']
        self.n_cells = int(math.ceil(self.density * math.pow(self.size, 2)))

        self.t_end = p['t_end']
        self.t_start = START_TIME
        self.dt = p['dt']
        h = p['h']
        self.h2 = h * h

    def run(self):
        S_e = np.random.rand(self.size, self.size)
        S_i = np.zeros((self.size, self.size), dtype=float)

        CELLS = np.zeros((self.size, self.size), dtype=int)

        for i in range(0, self.n_cells):
            idxI = randint(0, self.size - 1)
            idxJ = randint(0, self.size - 1)
            while CELLS[idxI][idxJ] == 1:
                idxI = randint(0, self.size - 1)
                idxJ = randint(0, self.size - 1)
            CELLS[idxI][idxJ] = 1

        A = CELLS * np.random.rand(self.size, self.size) * 100
        B = CELLS * np.random.rand(self.size, self.size) * 100
        C = CELLS * np.random.rand(self.size, self.size) * 100

        mA = CELLS * np.random.rand(self.size, self.size) * 100
        mB = CELLS * np.random.rand(self.size, self.size) * 100
        mC = CELLS * np.random.rand(self.size, self.size) * 100

        # A_series = np.zeros(int(t_end/dt), dtype=float)
        # S_e_series = np.zeros(int(t_end/dt), dtype=float)
        A_full = np.zeros((int(self.t_end / self.dt) + 1, self.size * self.size), dtype=float)

        t = 0
        k = 0
        step = 0

        A_full[step, :] = A.flatten('F')  # [A>0]

        a = np.arange(1, self.size, dtype=int)
        b = np.arange(0, self.size - 1, dtype=int)
        i = np.argsort(np.append(a, 0))
        j = np.argsort(np.append(self.size - 1, b))

        # merjenje casa
        timeMeasure = time.time()

        while t <= self.t_end:
            S_e_xx = []
            S_e_yy = []
            if self.periodic_bounds:
                # D1 * ([S_e(:,end),S_e(:,1:end-1)] + [S_e(:,2:end),S_e(:,1)] -2*S_e)/h2
                S_e_xx = self.D1 * (S_e[:, i] + S_e[:, j] - 2 * S_e) / self.h2
                # D1 * ([S_e(end,:);S_e(1:end-1,:)] + [S_e(2:end,:);S_e(1,:)] -2*S_e)/h2
                S_e_yy = self.D1 * (S_e[i, :] + S_e[j, :] - 2 * S_e) / self.h2
            else:
                X = np.zeros((self.size + 2, self.size + 2), dtype=float)
                X[0, 1:self.size + 1] = S_e[1, :]
                X[self.size + 1, 1:self.size + 1] = S_e[len(S_e) - 2, :]
                X[1:self.size + 1, 0] = S_e[:, 1]
                X[1:self.size + 1, self.size + 1] = S_e[:, len(S_e) - 2]
                X[1:self.size + 1, 1:self.size + 1] = S_e

                S_e_xx = self.D1 * (X[1:len(X) - 1, 0:len(X) - 2] + X[1:len(X) - 1, 2:len(X)] - 2 * S_e) / self.h2
                S_e_yy = self.D1 * (X[0:len(X) - 2, 1:len(X) - 1] + X[2:len(X), 1:len(X) - 1] - 2 * S_e) / self.h2

            D2S_e = S_e_xx + S_e_yy

            # Calculate dx/dt
            # [dmA, dmB, dmC, dA, dB, dC, dS_i, dS_e] = repressilator_S_ODE(CELLS, mA, mB, mC, A, B, C, S_i, S_e, alpha, alpha0, Kd, beta, delta_m, delta_p, n, kS0, kS1, kSe, kappa, eta)
            # time = np.linspace(0.0,100.0,1000)
            # vec = odeint(model, A.flatten('F'), time)

            dmA = CELLS * (self.alpha / (1 + np.power((C / self.Kd), self.n)) + self.alpha0 - self.delta_m * mA)
            dmB = CELLS * (self.alpha / (1 + np.power((A / self.Kd), self.n)) + self.alpha0 - self.delta_m * mB)
            dmC = CELLS * (self.alpha / (1 + np.power((B / self.Kd), self.n)) + self.alpha0 - self.delta_m * mC + (self.kappa * S_i) / (1 + S_i))

            dA = CELLS * (self.beta * mA - self.delta_p * A)
            dB = CELLS * (self.beta * mB - self.delta_p * B)
            dC = CELLS * (self.beta * mC - self.delta_p * C)

            dS_i = CELLS * (- self.kS0 * S_i + self.kS1 * A - self.eta * (S_i - S_e))
            dS_e = - self.kSe * S_e + CELLS * (self.eta * (S_i - S_e))

            dS_e = dS_e + D2S_e

            if self.borderfixed:
                # leave border as distrotion centers
                dS_e[0:self.size, 0] = 0
                dS_e[0:self.size, self.size - 1] = 0
                dS_e[0, 0:self.size] = 0
                dS_e[self.size - 1, 0:self.size] = 0

            mA = mA + self.dt * dmA
            mB = mB + self.dt * dmB
            mC = mC + self.dt * dmC
            A = A + self.dt * dA
            B = B + self.dt * dB
            C = C + self.dt * dC
            S_i = S_i + self.dt * dS_i
            S_e = S_e + self.dt * dS_e

            t = t + self.dt
            step = step + 1

            A_full[step, :] = A.flatten('F')  # [A>0]

        # izpis casa
        print("Porabljen cas: {} s.".format(time.time() - timeMeasure))

        # T = 0:dt:t_end-dt
        T = np.arange(0, self.t_end - self.dt, self.dt, dtype=int)

        # ------------------------------------------------------------------------------------------------------------ #
        # DOES IT OSCILLATE? from start time to end time
        result = my_utils.measure_osc(A_full[self.t_start:self.t_end, self.size * self.size - 20], T[self.t_start:self.t_end], 0.1)
        print(result)

        # ------------------------------------------------------------------------------------------------------------ #
        # DRAW GRAPH

        # TT = T.'
        # TMat = repmat(TT,1,n_cells);
        TMat = np.matlib.repmat(T, 1, self.n_cells)
        y = np.arange(0, self.n_cells, dtype=int)
        # yMat = repmat(y, numel(TT), 1); #//For plot3
        yMat = np.matlib.repmat(y, len(T), 1)

        fig, ax = plt.subplots()
        ax.plot(A_full[self.t_start:self.t_end, self.size * self.size - 1])
        plt.show()
        # ------------------------------------------------------------------------------------------------------------ #


if __name__ == "__main__":
    r = Repressilator_S_PDE()
    r.load_params()

    for i in range(0, 5):
        r.run()
