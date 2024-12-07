import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt

from dvr import compute_ground_energy

mpl.use('module://matplotlib-backend-sixel')

gauss_potential = lambda x: -4.0 * np.exp(-(x**2) / 4.0)

# For 2 Particles

Ls = [5.0, 7.0, 10.0]
Ns = range(2, 16, 2)

energies = np.array(
    [[compute_ground_energy(2, L, gauss_potential, N) for N in Ns] for L in Ls]
)

E_expected = -3.094

fig = plt.figure()

for i, l in enumerate(Ls):
    plt.plot(Ns, energies[i], label=f'Box Size {l}')

plt.axhline(E_expected)
plt.xlabel('# Partitions')
plt.ylabel('Estimate of $E_0$')
plt.title('Energy of the Estimated $E_0$ for the Two Particle Case')
plt.legend()

fig.savefig('./plots/2_particle_benchmark.png')
fig.show()


# For 3 Particles

Ls = [5.0, 7.0, 10.0]
Ns = range(2, 16, 2)

energies = np.array(
    [[compute_ground_energy(3, L, gauss_potential, N) for N in Ns] for L in Ls]
)

E_expected = -9.738

fig = plt.figure()

for i, l in enumerate(Ls):
    plt.plot(Ns, energies[i], label=f'Box Size {l}')

plt.axhline(E_expected)
plt.xlabel('# Partitions')
plt.ylabel('Estimate of $E_0$')
plt.title('Energy of the Estimated $E_0$ for the Three Particle Case')
plt.legend()

fig.savefig('./plots/3_particle_benchmark.png')
fig.show()


# For 4 Particles

Ls = [5.0, 7.0, 10.0]
Ns = range(2, 16, 2)

energies = np.array(
    [[compute_ground_energy(4, L, gauss_potential, N) for N in Ns] for L in Ls]
)

E_expected = -20.04

fig = plt.figure()

for i, l in enumerate(Ls):
    plt.plot(Ns, energies[i], label=f'Box Size {l}')

plt.axhline(E_expected)
plt.xlabel('# Partitions')
plt.ylabel('Estimate of $E_0$')
plt.title('Energy of the Estimated $E_0$ for the Four Particle Case')
plt.legend()

fig.savefig('./plots/4_particle_benchmark.png')
fig.show()
