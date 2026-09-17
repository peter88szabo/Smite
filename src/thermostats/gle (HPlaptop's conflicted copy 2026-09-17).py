import os

import numpy as np

from sampling.random_seed import sampling_generator


def _read_gle_matrix(path, expected_ns=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GLE matrix file not found: {path}")

    with open(path, "r") as handle:
        rows = [
            line.split()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not rows:
        raise ValueError(f"GLE matrix file is empty: {path}")

    first_row = rows[0]
    if len(first_row) == 1 and len(rows) > 1:
        ns = int(float(first_row[0]))
        matrix = np.array([[float(value) for value in row] for row in rows[1:]], dtype=float)
        expected_shape = (ns + 1, ns + 1)
        if matrix.shape != expected_shape:
            raise ValueError(f"{path} has shape {matrix.shape}, expected {expected_shape}")
    else:
        matrix = np.array([[float(value) for value in row] for row in rows], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{path} must contain a square GLE matrix")
        ns = matrix.shape[0] - 1

    if expected_ns is not None and ns != expected_ns:
        raise ValueError(f"GLE matrix size mismatch: {path} has ns={ns}, expected ns={expected_ns}")

    return ns, matrix


def matrix_exp(matrix, taylor_order=15, scale_power=15):
    coeff = [1.0]
    for i in range(1, taylor_order + 1):
        coeff.append(coeff[-1] / float(i))

    scaled = matrix / (2.0 ** scale_power)
    expm = np.eye(matrix.shape[0]) * coeff[taylor_order]
    for power in range(taylor_order - 1, -1, -1):
        expm = scaled @ expm
        expm += np.eye(matrix.shape[0]) * coeff[power]

    for _ in range(scale_power):
        expm = expm @ expm

    return expm


def cholesky_stabilized(matrix):
    """Return a factor ``S`` satisfying ``S @ S.T == matrix``.

    GLE covariance matrices can be positive semidefinite, so a conventional
    Cholesky factorization is too strict when a decoupled auxiliary variable
    gives an exact zero eigenvalue. Use a symmetric eigensystem, reject
    genuinely indefinite inputs, and clip only round-off-sized negatives.

    The historical public name is retained for compatibility even though the
    returned square root is not necessarily triangular.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("GLE covariance matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("GLE covariance matrix must contain only finite values")

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1.0e-12 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(
            "GLE covariance matrix is not positive semidefinite: "
            f"minimum eigenvalue={float(np.min(eigenvalues)):.6e}"
        )

    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors * np.sqrt(eigenvalues)[None, :]


class GLEThermostat:
    def __init__(self, dt, wopt, kt, ndim, a_file="GLE-A", c_file="GLE-C", rng=None):
        if dt <= 0.0:
            raise ValueError("GLE timestep must be positive")
        if wopt <= 0.0:
            raise ValueError("GLE wopt must be positive")
        if kt <= 0.0:
            raise ValueError("GLE temperature must be positive")

        self.rng = rng if rng is not None else sampling_generator()
        self.ns, gA = _read_gle_matrix(a_file)
        gA = gA * wopt

        if c_file is not None and os.path.exists(c_file):
            _, gC = _read_gle_matrix(c_file, expected_ns=self.ns)
        else:
            gC = np.eye(self.ns + 1) * kt

        self.gT = matrix_exp(-dt * gA)
        noise_covariance = gC - self.gT @ gC @ self.gT.T
        noise_covariance = 0.5 * (noise_covariance + noise_covariance.T)
        self.gS = cholesky_stabilized(noise_covariance)

        c_chol = cholesky_stabilized(gC)
        noise = self.rng.normal(size=(ndim, self.ns + 1))
        self.gp = noise @ c_chol.T

    def step(self, p, wmass, nfix=0):
        p = np.array(p, copy=True)
        removed_dof = int(nfix)
        if removed_dof < 0 or removed_dof >= len(p):
            raise ValueError("GLE nfix must satisfy 0 <= nfix < 3N")
        # nfix is a count, not a suffix of Cartesian coordinates to freeze.
        # COM and holonomic constraints are projected after the OU update.
        active = np.arange(len(p))

        mass_sqrt = np.sqrt(np.asarray(wmass, dtype=float)[active])
        self.gp[active, 0] = p[active] / mass_sqrt

        deterministic = self.gp[active, :] @ self.gT.T
        noise = self.rng.normal(size=(len(active), self.ns + 1))
        self.gp[active, :] = deterministic + noise @ self.gS.T

        p[active] = self.gp[active, 0] * mass_sqrt
        return p


def make_gle_thermostat(dt, wopt, kt, ndim, a_file="GLE-A", c_file="GLE-C", rng=None):
    return GLEThermostat(dt, wopt, kt, ndim, a_file=a_file, c_file=c_file, rng=rng)


def thermo_gle(nfix, p, wmass, state):
    if state is None:
        raise ValueError("GLE thermostat state has not been initialized")
    return state.step(p, wmass, nfix=nfix)
