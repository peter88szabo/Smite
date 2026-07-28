import os

import numpy as np


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
    n = matrix.shape[0]
    lmat = np.zeros_like(matrix, dtype=float)
    dmat = np.zeros_like(matrix, dtype=float)

    for i in range(n):
        lmat[i, i] = 1.0
        for j in range(i):
            value = matrix[i, j]
            for k in range(j):
                value -= lmat[i, k] * lmat[j, k] * dmat[k, k]
            lmat[i, j] = value / dmat[j, j] if dmat[j, j] != 0.0 else 0.0

        value = matrix[i, i]
        for k in range(i):
            value -= lmat[i, k] * lmat[i, k] * dmat[k, k]
        dmat[i, i] = np.sqrt(value) if value >= 0.0 else 0.0

    return lmat @ dmat


class GLEThermostat:
    def __init__(self, dt, wopt, kt, ndim, a_file="GLE-A", c_file="GLE-C", rng=None):
        if dt <= 0.0:
            raise ValueError("GLE timestep must be positive")
        if wopt <= 0.0:
            raise ValueError("GLE wopt must be positive")
        if kt <= 0.0:
            raise ValueError("GLE temperature must be positive")

        self.rng = rng if rng is not None else np.random.default_rng()
        self.ns, gA = _read_gle_matrix(a_file)
        gA = gA * wopt

        if c_file is not None and os.path.exists(c_file):
            _, gC = _read_gle_matrix(c_file, expected_ns=self.ns)
        else:
            gC = np.eye(self.ns + 1) * kt

        self.gT = matrix_exp(-dt * gA)
        self.gS = cholesky_stabilized(gC - self.gT @ gC @ self.gT.T)

        c_chol = cholesky_stabilized(gC)
        noise = self.rng.normal(size=(ndim, self.ns + 1))
        self.gp = noise @ c_chol.T

    def step(self, p, wmass, nfix=0):
        p = np.array(p, copy=True)
        nactive = len(p) - int(nfix)
        if nactive < 0:
            raise ValueError("GLE nfix cannot exceed the number of momentum components")
        active = np.arange(nactive)
        if len(active) == 0:
            return p

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
