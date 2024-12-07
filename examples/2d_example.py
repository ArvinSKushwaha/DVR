from typing import cast

import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

from dvr import System

mpl.use('module://matplotlib-backend-sixel')

gauss_potential = lambda x: -4.0 * np.exp(-(x**2) / 4.0)
N = 20

s = System(3, 6.0, N)
H = s.make_hamiltonian(gauss_potential)

xs = s.get_positions()

eigvals, eigvecs = np.linalg.eigh(H.todense())

fig = plt.figure()
state = eigvecs[:, 0].real
state /= np.linalg.norm(state) * (np.ptp(xs) ** 2 / state.size)

M = 100
XX, YY = np.mgrid[xs.min() : xs.max() : M * 1j, xs.min() : xs.max() : M * 1j]
ZZ = griddata(
    np.stack(np.meshgrid(xs, xs), axis=-1).reshape((-1, 2)),
    state,
    (XX, YY),
    method='cubic',
)
plt.contourf(XX, YY, ZZ)
plt.colorbar()

plt.xlabel('$x_1$')
plt.ylabel('$x_1$')
plt.title('Ground State of 3-Particle System with Gaussian Potential')

fig.show()
fig.savefig('./plots/2d_example.png')
