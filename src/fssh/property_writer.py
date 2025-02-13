import numpy as np


class PropertyWriter:
    def __init__(self, filename='properties.csv'):
        self.filename = filename
        self.file = open(self.filename, 'w', buffering=1)
        self.header = 0

    def write(self, epot: float, c: np.ndarray, step: int, dt_class: float, active_state: int, coupling: np.ndarray):
        rho = self._calculate_density_matrix(c)
        d_max = np.real(coupling.max())

        if self.header == 0:
            header = self._create_header(rho)
            self.file.write(f"step,time,{','.join(header)},active_state,d_max,epot\n")
            self.header = 1

        self.file.write(
            f"{step},{step * dt_class},{','.join([str(i) for i in rho.ravel().tolist()])},{active_state},{d_max},{epot}\n")

        self.file.flush()

    def _calculate_density_matrix(self, c: np.ndarray):
        """This is an implementation that is not appropriate for long dynamics simulations"""
        rho = np.zeros((c.shape[0], c.shape[0]), dtype=float)

        for i in range(c.shape[0]):
            for j in range(c.shape[0]):
                rij = np.conj(c[i]) * c[j]
                rho[i, j] = abs(rij)

        return rho

    def _create_header(self, rho):
        header = []
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                header.append(f'c{i}{j}')

        return header

    def close(self):
        self.file.close()
