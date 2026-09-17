"""Conservation and convergence regressions for rigid-body fallback dynamics."""

import numpy as np
import pytest

from core.molecule import Molecule
from dynamics.integrator_driver import apply_integrator


def _body(reference, mass, omega, algorithm="rattle"):
    reference = np.array(reference, dtype=float)
    reference -= np.average(reference, weights=mass, axis=0)
    p = np.asarray(mass)[:, None] * np.cross(omega, reference)
    molecule = Molecule(["X"] * len(mass), np.array(mass), reference.ravel(), p.ravel())
    linear = np.linalg.matrix_rank(reference) == 1
    molecule.add_rigid_constraint_group(
        range(len(mass)), degrees_of_freedom_removed=3 * len(mass) - (5 if linear else 6)
    )
    molecule.nfix = 3 * len(mass) - (5 if linear else 6)
    molecule.prepare_rigid_constraints(algorithm=algorithm)
    molecule.qchem = {"wfu": False}
    assert molecule._constraint_solver.singular_group_indices == {0}
    return molecule


def _energy(molecule):
    return 0.5 * np.sum(molecule.p**2 / molecule.wmass)


def _angular(molecule):
    return np.cross(molecule.q.reshape(-1, 3), molecule.p.reshape(-1, 3)).sum(axis=0)


@pytest.mark.parametrize("algorithm", ["shake", "rattle"])
@pytest.mark.parametrize("linear", [False, True])
def test_free_symmetric_and_linear_rotors_have_no_damping(monkeypatch, algorithm, linear):
    monkeypatch.setattr("integrators.rattle.force_calc", lambda _qc, q, _atoms: np.zeros_like(q))
    reference = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]] if linear else [
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]
    ]
    body = _body(reference, np.ones(len(reference)), [0, 0, 1], algorithm)
    velocity = np.array([.2, -.1, .4])
    body.p += body.wmass * np.tile(velocity, len(reference))
    q0, p0 = body.q.copy(), body.p.copy()
    energy0, angular0 = _energy(body), _angular(body)
    for _ in range(200):
        apply_integrator(body, "verlet", dt=0.05)
    assert _energy(body) == pytest.approx(energy0, rel=1e-10)
    np.testing.assert_allclose(_angular(body), angular0, atol=1e-10)
    rotation = np.array([[np.cos(10), -np.sin(10), 0], [np.sin(10), np.cos(10), 0], [0, 0, 1]])
    np.testing.assert_allclose(body.q.reshape(-1, 3), q0.reshape(-1, 3) @ rotation.T + 10 * velocity, atol=1e-10)
    np.testing.assert_allclose(body.p.reshape(-1, 3), (p0.reshape(-1, 3) - velocity) @ rotation.T + velocity, atol=1e-10)


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("force_driven", [False, True])
def test_asymmetric_rigid_body_is_reversible_and_second_order(monkeypatch, large, force_driven):
    if large:
        t = np.linspace(0, 3 * np.pi, 40)
        reference = np.column_stack((np.cos(t), 2 * np.sin(t), t / 3))
    else:
        reference = np.array([[-2, -1, 0], [2, -1, 0], [1, 1, 0], [-1, 1, 0]])
    mass = np.linspace(1, 2, len(reference))
    stiffness = 0.3 if force_driven else 0.0
    monkeypatch.setattr("integrators.rattle.force_calc", lambda _qc, q, _atoms: -stiffness * q)
    errors = []
    for dt in (0.04, 0.02):
        body = _body(reference, mass, [0.3, -0.4, 0.8])
        q0, p0 = body.q.copy(), body.p.copy()
        angular0 = _angular(body)
        energy0 = _energy(body) + 0.5 * stiffness * np.dot(body.q, body.q)
        max_error = 0
        for _ in range(round(4 / dt)):
            apply_integrator(body, "verlet", dt=dt)
            energy = _energy(body) + 0.5 * stiffness * np.dot(body.q, body.q)
            max_error = max(max_error, abs(energy - energy0))
        errors.append(max_error)
        np.testing.assert_allclose(_angular(body), angular0, rtol=1e-9, atol=1e-9)
        assert body._constraint_solver.maximum_position_error(body.q) < 1e-10
        assert body._constraint_solver.maximum_velocity_error(body.q, body.p, body.wmass) < 1e-10
        for _ in range(round(4 / dt)):
            apply_integrator(body, "verlet", dt=-dt)
        np.testing.assert_allclose(body.q, q0, atol=1e-9)
        np.testing.assert_allclose(body.p, p0, atol=1e-9)
    assert errors[0] / errors[1] == pytest.approx(4, rel=0.08)
