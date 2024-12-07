import numpy as np

from matplotlib import pyplot as plt

from dvr import System

gauss_potential = lambda x: -4.0 * np.exp(-(x**2) / 4.0)
s = System(2, 10.0, 500)
H = s.make_hamiltonian(gauss_potential)

xs = s.get_positions()

eigvals, eigvecs = np.linalg.eigh(H.todense())

fig = plt.figure()

k = 3
for i in range(k):
    plt.axhline(eigvals[i], linestyle=':')

    state = eigvecs[:, i].real
    state /= np.linalg.norm(state) * (np.ptp(xs) / state.size)

    plt.plot(xs, state * 0.1 + eigvals[i])

plt.plot(xs, gauss_potential(xs), 'k')

plt.xlabel('$x_1$')
plt.ylabel('$E$')
plt.title('Ground and Excited States of 2-Particle System with Gaussian Potential')

fig.savefig('./plots/1d_example.png')
