from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt

from scipy.sparse import bsr_array, dia_array, diags_array, eye_array, kron
from scipy.sparse import linalg as splin


def kron_seq(*arrays: Any, **kwargs: Any) -> Any:
    array = np.eye(1)
    for arr in arrays:
        array = kron(array, arr, **kwargs)
    return array


def make_del_operator(n: int, L: float) -> npt.NDArray[np.complex128]:
    k = np.arange(n)[:, None]
    l = k.T
    k_minus_l = k - l

    operator = np.zeros((n, n), dtype=np.complex128)
    np.fill_diagonal(operator, -1j)

    mask = k_minus_l != 0
    np.place(
        operator,
        mask,
        (-1.0) ** (k_minus_l[mask])
        * np.exp(-1j * np.pi * k_minus_l[mask] / n)
        / np.sin(np.pi * k_minus_l[mask] / n),
    )

    return cast(
        npt.NDArray[np.complex128],
        operator * (np.pi / L),
    )


@dataclass
class System:
    n_particles: int
    box_length: float
    n_mesh: int

    def make_potential_matrix(
        self,
        V_central: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> dia_array:
        n = self.n_mesh
        N = self.n_mesh ** (self.n_particles - 1)

        indices = np.arange(N)

        positions = []
        for i in range(self.n_particles - 1):
            positions.append(((indices % n) - n // 2) * self.box_length / n)
            indices //= n

        positions.reverse()
        potential = np.zeros(N)

        for i in range(self.n_particles - 1):
            potential += V_central(np.abs(positions[i]))

            for j in range(i):
                displacement = np.abs(positions[i] - positions[j])
                displacement = np.where(
                    displacement > self.box_length / 2,
                    self.box_length - displacement,
                    displacement,
                )
                potential += V_central(displacement)

        return diags_array(potential, dtype=np.complex128)

    def make_kinetic_matrix(self) -> bsr_array:
        N = self.n_mesh ** (self.n_particles - 1)
        kinetic_energy = bsr_array((N, N))

        del_operator = make_del_operator(self.n_mesh, self.box_length)
        del2_operator = del_operator @ del_operator

        for i in range(self.n_particles - 1):
            for j in range(i + 1):
                if i == j:
                    kinetic_energy += kron_seq(
                        eye_array((self.n_mesh) ** j),
                        del2_operator,
                        eye_array(self.n_mesh ** (self.n_particles - j - 2)),
                    )
                else:
                    kinetic_energy += kron_seq(
                        eye_array((self.n_mesh) ** j),
                        del_operator,
                        eye_array((self.n_mesh) ** (i - j - 1)),
                        del_operator,
                        eye_array(self.n_mesh ** (self.n_particles - i - 2)),
                    )
        mu = 1.0 / 2.0  # since we have all particles with mass 1
        kinetic_energy = kinetic_energy * -1.0 / (2.0 * mu)
        kinetic_energy.eliminate_zeros()
        return kinetic_energy.tobsr()

    def make_hamiltonian(
        self,
        V_central: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> bsr_array:
        return self.make_kinetic_matrix() + self.make_potential_matrix(V_central)

    def get_positions(self) -> npt.NDArray[np.float64]:
        return (
            (np.arange(self.n_mesh) - self.n_mesh // 2) * self.box_length / self.n_mesh
        )


def compute_ground_energy(
    n_particles: int,
    L: float,
    V_central: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    mesh_size: int = 5,
) -> float:
    system = System(n_particles, L, mesh_size)
    H = system.make_hamiltonian(V_central)

    if (H.count_nonzero() > 0.2 * H.size) and H.size < 10_000**2:
        return np.linalg.eigvalsh(H.todense()).min()
    else:
        return splin.eigsh(H, k=1, which='SA')[0][0]
