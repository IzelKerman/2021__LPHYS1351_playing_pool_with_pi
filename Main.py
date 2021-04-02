import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class System:
    def __init__(self, M, x, v):
        self.M = M
        self.x = x
        self.v = v
        self.M_12 = self.M[0] + self.M[1]
        self.M_23 = self.M[1] + self.M[2]
        self.mu_12 = (self.M[0] - self.M[1]) / self.M_12
        self.mu_23 = (self.M[1] - self.M[2]) / self.M_23

    def change_masses(self, M):
        self.__init__(M, self.x, self.v)

    def next_collision(self):   # oui c'est pas optimisé, osef
        if self.v[0] < 0 and self.x[0] != 0:
            dt_a = - self.x[0] / self.v[0]
        else:
            dt_a = math.inf
        if self.v[1] - self.v[0] < 0:
            dt_b = (self.x[0] - self.x[1]) / (self.v[1] - self.v[0])
        else:
            dt_b = math.inf
        if self.v[2] - self.v[1] < 0:
            dt_c = (self.x[1] - self.x[2]) / (self.v[2] - self.v[1])
        else:
            dt_c = math.inf
        if abs(dt_a - dt_b) < 1e-16 and abs(dt_a - dt_c) < 1e-16:
            print("woops, triple collision occured")
            return [0, 0, 0], [0, 0, 0]  # error, triple collision, can't compute the velocities
        elif abs(dt_a - dt_c) < 1e-16 and dt_c < dt_a:
            x_0 = self.x[0] + self.v[0] * dt_b
            print("woops, triple collision occured")
            return [x_0, x_0, x_0], [0, 0, 0]  # error, triple collision, can't compute the velocities
        elif abs(dt_a - dt_b) < 1e-16 and dt_b < dt_c:
            print("woops, triple collision occured [kinda]")
            return [0, 0, self.x[2] + self.v[2] * dt_a], [0, 0, 0]  # error, triple collision [with the wall], can't compute the velocities
        elif abs(dt_a - dt_c) < 1e-16 and dt_c < dt_b:
            x_0 = self.x[1] + self.v[1] * dt_a
            return [0, x_0, x_0], [-self.v[0], self.mu_23 * self.v[1] + 2 * self.M[2] * self.v[2] / self.M_23, 2 * self.M[1] * self.v[1] / self.M_23 - self.mu_23 * self.v[2]]
        elif dt_b > dt_a < dt_c:
            return [0, self.x[1] + self.v[1] * dt_a, self.x[2] + self.v[2] * dt_a], [-self.v[0], self.v[1], self.v[2]]
        elif dt_a > dt_b < dt_c:
            return [self.x[0] + self.v[0] * dt_b, self.x[1] + self.v[1] * dt_b, self.x[2] + self.v[2] * dt_b], [self.mu_12 * self.v[0] + 2 * self.M[1] * self.v[1] / self.M_12, 2 * self.M[0] * self.v[0] / self.M_12 - self.mu_12 * self.v[1], self.v[2]]
        else:
            return [self.x[0] + self.v[0] * dt_c, self.x[1] + self.v[1] * dt_c, self.x[2] + self.v[2] * dt_c], [self.v[0], self.mu_23 * self.v[1] + 2 * self.M[2] * self.v[2] / self.M_23, 2 * self.M[1] * self.v[1] / self.M_23 - self.mu_23 * self.v[2]]

    def compute_number_collisions(self, max=None):
        n = 0
        if max is None:
            while not 0 <= self.v[0] <= self.v[1] <= self.v[2]:
                self.x, self.v = self.next_collision()
                n += 1
        else:
            while not (0 <= self.v[0] <= self.v[1] <= self.v[2] or n > max):
                self.x, self.v = self.next_collision()
                n += 1
        return n


def plot_collision_domain(x_0, ax, domain, N, z_domain=None, log=False, to_print="coucou"):
    """
    x_0 : 1d array with the initial position of each ball.     ex: x_0 = [1, 2, 3] ; first ball at x = 1, second at y = 2, third at z = 3
    ax : an ax from matplotlib.
    domain : 2d array representing the domain of the plot.     ex: [[0, 1], [2, 3]] will give a plot for log(m_2/m_1) in [0, 1] and log(m_3/m_1) in [2, 3]
    N : 1d array the number of point on wich we compute the number of collisions. (precision of the plot)
    """
    x = np.array([10 ** (((N[0] - i) * domain[0][0] + i * domain[0][1]) / N[0]) for i in range(0, N[0] + 1)])
    y = np.array([10 ** (((N[1] - i) * domain[1][0] + i * domain[1][1]) / N[1]) for i in range(0, N[1] + 1)])
    X, Y = np.meshgrid(x, y)
    Z = np.copy(X)
    t = time.process_time()
    try:
        z_max = z_domain[1]
    except:
        z_max = None
    for i, z_i in enumerate(Z):
        for j, z_ij in enumerate(z_i):
            syst = System([1, X[i, j], Y[i, j]], x_0, [0, 0, -1])
            Z[i, j] = syst.compute_number_collisions(max=z_max)
            if time.process_time() - t > 1:
                t = time.process_time()
                print("{:.3f}".format((i * (N[1] + 1) + j) / ((N[0] + 1) * (N[1] + 1))))
    ax.set_aspect(1)
    lX = np.log10(X)
    lY = np.log10(Y)
    if not log:
        try:
            z_min = z_domain[0]
            z_max = z_domain[1]
        except:
            z_max = np.amax(Z)
            z_min = np.amin(Z)
        levels = [z_min - 0.5 + i for i in range(int(z_max) - int(z_min) + 2)]
        ticks_levels = [int(z_min) + i for i in range(int(z_max) - int(z_min) + 1)]
        lol = ax.contourf(lX, lY, Z, levels, cmap='plasma')
        ax.contour(lX, lY, Z, ticks_levels, colors='black', linewidths=0.4)
        ticks_levels_2 = [k for k in [3 + 3 * i for i in range(22)]]
        fig.colorbar(lol, ticks=ticks_levels_2)
    else:
        lZ = np.log10(Z)
        lol = ax.contourf(lX, lY, lZ, 100, cmap='plasma')
        ax.contour(lX, lY, lZ, 101, colors='black', linewidths=0.4)
        fig.colorbar(lol)
    ax.set_xlabel('$\log(m_2 / m_1)$')
    ax.set_ylabel('$\log(m_3 / m_1)$')
    ax.set_title(to_print)
    ax.set_xlim(domain[0])
    ax.set_ylim(domain[1])


def centered_domain(c, a):
    return [[c[0] - a[0], c[0] + a[0]], [c[1] - a[1], c[1] + a[1]]]


def u(k):
    return 1 / np.tan(np.pi / (k + 3)) ** 2


if __name__ == "__main__":
    N = [400, 400]
    x_0 = [1, 2, 3]
    c = [0.5, 0.5]
    r = 1e0
    domain = centered_domain(c, [r, r])
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    plot_collision_domain(x_0, ax, domain, N, log=False, to_print="Number of collisions for $d=${}".format(x_0[1]))
    plt.show()
