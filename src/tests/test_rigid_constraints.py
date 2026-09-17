import numpy as np
import pytest

from core.collision import Collision
from core.fragment import Fragment
from core.molecule import Molecule
from dynamics.constraints import RigidConstraintSolver
from dynamics.integrator_driver import apply_integrator
from dynamics.thermostat_driver import apply_thermostat


def _distance(q, atom_i, atom_j):
    xyz = np.asarray(q, dtype=float).reshape((-1, 3))
    return np.linalg.norm(xyz[atom_i] - xyz[atom_j])


def _linear_momentum(p):
    return np.asarray(p, dtype=float).reshape((-1, 3)).sum(axis=0)


def _angular_momentum(q, p):
    xyz = np.asarray(q, dtype=float).reshape((-1, 3))
    momentum = np.asarray(p, dtype=float).reshape((-1, 3))
    return np.sum(np.cross(xyz, momentum), axis=0)


def _diatom_solver(position_tolerance=1.0e-12, velocity_tolerance=1.0e-12):
    group = {
        "atom_indices": np.array([0, 1]),
        "reference_positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "degrees_of_freedom_removed": 1,
    }
    return RigidConstraintSolver.from_rigid_groups(
        [group],
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        max_iterations=200,
    )


def test_shake_restores_distance_and_preserves_center_of_mass():
    solver = _diatom_solver()
    masses = np.array([2.0, 5.0])
    wmass = np.repeat(masses, 3)
    q_old = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    q_trial = np.array([0.04, 0.02, 0.0, 1.10, -0.03, 0.01])
    p_drift = np.array([0.2, -0.1, 0.0, -0.4, 0.3, 0.1])
    center_before = np.average(q_trial.reshape((-1, 3)), axis=0, weights=masses)

    q_new, _p_new = solver.shake(q_old, q_trial, p_drift, dt=0.2, wmass=wmass)

    center_after = np.average(q_new.reshape((-1, 3)), axis=0, weights=masses)
    assert _distance(q_new, 0, 1) == pytest.approx(1.0, rel=0.0, abs=2.0e-12)
    np.testing.assert_allclose(center_after, center_before, rtol=0.0, atol=2.0e-14)
    assert solver.maximum_position_error(q_new) <= solver.position_tolerance


def test_rattle_enforces_tangent_velocity_and_preserves_total_impulses():
    solver = _diatom_solver()
    q = np.array([0.0, 0.0, 0.0, 0.6, 0.8, 0.0])
    p_trial = np.array([0.5, -0.7, 0.2, -0.3, 0.9, -0.4])
    wmass = np.repeat(np.array([2.0, 5.0]), 3)
    linear_before = _linear_momentum(p_trial)
    angular_before = _angular_momentum(q, p_trial)

    p_new = solver.rattle(q, p_trial, wmass)

    assert solver.maximum_velocity_error(q, p_new, wmass) <= solver.velocity_tolerance
    np.testing.assert_allclose(_linear_momentum(p_new), linear_before, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(_angular_momentum(q, p_new), angular_before, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("constraint_algorithm", ["shake", "rattle"])
def test_rigid_diatom_stays_rigid_during_force_driven_propagation(
    monkeypatch, constraint_algorithm
):
    molecule = Molecule(
        atoms=["X", "Y"],
        mass=np.array([2.0, 3.0]),
        q_ini=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        p_ini=np.array([0.0, 0.8, 0.0, 0.0, -0.2, 0.4]),
        nfix=1,
    )
    molecule.add_rigid_constraint_group([0, 1], degrees_of_freedom_removed=1)
    molecule.prepare_rigid_constraints(
        algorithm=constraint_algorithm,
        position_tolerance=1.0e-11,
        velocity_tolerance=1.0e-11,
    )
    molecule.qchem = {"wfu": False}

    def stretching_force(_qcinput, q, _atoms):
        xyz = np.asarray(q).reshape((-1, 3))
        direction = xyz[1] - xyz[0]
        direction /= np.linalg.norm(direction)
        return np.concatenate((-0.7 * direction, 1.3 * direction))

    monkeypatch.setattr("integrators.rattle.force_calc", stretching_force)

    for _ in range(250):
        apply_integrator(molecule, "verlet", dt=0.01)

    assert _distance(molecule.q, 0, 1) == pytest.approx(1.0, rel=0.0, abs=2.0e-10)
    assert molecule._constraint_solver.maximum_position_error(molecule.q) <= 1.0e-11
    if constraint_algorithm == "rattle":
        assert molecule._constraint_solver.maximum_velocity_error(
            molecule.q, molecule.p, molecule.wmass
        ) <= 1.0e-11


def test_free_rattle_rotation_conserves_linear_and_angular_momentum(monkeypatch):
    molecule = Molecule(
        atoms=["X", "Y"],
        mass=np.array([2.0, 3.0]),
        q_ini=np.array([-0.6, 0.0, 0.0, 0.4, 0.0, 0.0]),
        p_ini=np.array([0.0, 0.25, 0.0, 0.0, -0.25, 0.0]),
        nfix=1,
    )
    molecule.add_rigid_constraint_group([0, 1], degrees_of_freedom_removed=1)
    molecule.prepare_rigid_constraints(
        algorithm="rattle",
        position_tolerance=1.0e-12,
        velocity_tolerance=1.0e-12,
    )
    molecule.qchem = {"wfu": False}
    monkeypatch.setattr(
        "integrators.rattle.force_calc", lambda _qcinput, q, _atoms: np.zeros_like(q)
    )
    initial_linear = _linear_momentum(molecule.p)
    initial_angular = _angular_momentum(molecule.q, molecule.p)
    initial_kinetic = 0.5 * np.sum(molecule.p * molecule.p / molecule.wmass)
    kinetic_energies = []

    for _ in range(2000):
        apply_integrator(molecule, "verlet", dt=0.01)
        kinetic_energies.append(0.5 * np.sum(molecule.p * molecule.p / molecule.wmass))

    np.testing.assert_allclose(_linear_momentum(molecule.p), initial_linear, atol=2.0e-12)
    np.testing.assert_allclose(_angular_momentum(molecule.q, molecule.p), initial_angular, atol=2.0e-10)
    assert _distance(molecule.q, 0, 1) == pytest.approx(1.0, rel=0.0, abs=2.0e-12)
    assert max(abs(np.asarray(kinetic_energies) - initial_kinetic)) < 2.0e-7


def test_collision_keeps_rigid_groups_separate():
    fragment_a = Fragment.Diatom_Init(
        "A", ["H", "H"], req=0.74, rigid=True, random_rot=False
    )
    fragment_b = Fragment.Diatom_Init(
        "B", ["Cl", "Cl"], req=1.99, rigid=True, random_rot=False
    )
    collision = Collision(
        fragment_a,
        fragment_b,
        {"qchem": "Psi4", "functional": "hf"},
    )
    solver = collision.prepare_rigid_constraints()

    pairs = {(item.atom_i, item.atom_j) for item in solver.constraints}
    assert pairs == {(0, 1), (2, 3)}
    assert collision.nfix == 2
    assert solver.degrees_of_freedom_removed == 2


def test_planar_rigid_body_uses_stable_projection_for_out_of_plane_forces(monkeypatch):
    q_reference = np.array(
        [
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0,
        ]
    )
    molecule = Molecule(
        atoms=["X"] * 4,
        mass=np.array([1.0, 2.0, 3.0, 4.0]),
        q_ini=q_reference,
        p_ini=np.zeros(12),
        nfix=6,
    )
    molecule.add_rigid_constraint_group(range(4), degrees_of_freedom_removed=6)
    solver = molecule.prepare_rigid_constraints(
        algorithm="rattle",
        position_tolerance=1.0e-11,
        velocity_tolerance=1.0e-11,
    )
    molecule.qchem = {"wfu": False}
    reference_distances = {
        (i, j): _distance(q_reference, i, j)
        for i in range(4)
        for j in range(i + 1, 4)
    }

    assert solver.singular_group_indices == {0}

    def out_of_plane_force(_qcinput, _q, _atoms):
        return np.array(
            [
                0.0, 0.0, 0.9,
                0.0, 0.0, -0.4,
                0.0, 0.0, 0.2,
                0.0, 0.0, -0.7,
            ]
        )

    monkeypatch.setattr("integrators.rattle.force_calc", out_of_plane_force)
    for _ in range(100):
        apply_integrator(molecule, "leapfrog", dt=0.01)

    for pair, reference_distance in reference_distances.items():
        assert _distance(molecule.q, *pair) == pytest.approx(
            reference_distance, rel=0.0, abs=3.0e-11
        )
    assert solver.maximum_position_error(molecule.q) <= 1.0e-11
    assert solver.maximum_velocity_error(molecule.q, molecule.p, molecule.wmass) <= 1.0e-11


def test_large_rigid_group_uses_sparse_linear_scaling_projection():
    natom = 40
    parameter = np.linspace(0.0, 3.0 * np.pi, natom)
    reference = np.column_stack(
        (np.cos(parameter), np.sin(parameter), 0.15 * parameter)
    )
    masses = np.linspace(1.0, 3.0, natom)
    group = {
        "atom_indices": np.arange(natom),
        "reference_positions": reference,
        "degrees_of_freedom_removed": 3 * natom - 6,
    }
    solver = RigidConstraintSolver.from_rigid_groups(
        [group], position_tolerance=1.0e-11, velocity_tolerance=1.0e-11
    )
    deformation = np.column_stack(
        (
            0.01 * np.sin(2.0 * parameter),
            -0.02 * np.cos(3.0 * parameter),
            0.015 * np.sin(5.0 * parameter),
        )
    )
    q_trial = (reference + deformation + np.array([2.0, -1.0, 0.5])).ravel()
    momentum = np.column_stack(
        (
            np.sin(parameter),
            np.cos(2.0 * parameter),
            np.sin(3.0 * parameter),
        )
    ).ravel()
    wmass = np.repeat(masses, 3)

    q_new, p_drift = solver.shake(
        reference.ravel(), q_trial, momentum, dt=0.1, wmass=wmass
    )
    p_new = solver.rattle(q_new, p_drift, wmass)
    reference_distances = np.linalg.norm(
        reference[:, None, :] - reference[None, :, :], axis=2
    )
    new_xyz = q_new.reshape((-1, 3))
    new_distances = np.linalg.norm(
        new_xyz[:, None, :] - new_xyz[None, :, :], axis=2
    )

    assert solver.singular_group_indices == {0}
    assert len(solver.constraints) <= 3 * natom - 6
    np.testing.assert_allclose(new_distances, reference_distances, rtol=0.0, atol=2.0e-12)
    assert solver.maximum_velocity_error(q_new, p_new, wmass) <= 1.0e-11


def test_rattle_is_reapplied_after_thermostat_momentum_change(monkeypatch):
    molecule = Molecule(
        atoms=["X", "Y"],
        mass=np.array([2.0, 3.0]),
        q_ini=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        p_ini=np.zeros(6),
        nfix=1,
    )
    molecule.add_rigid_constraint_group([0, 1], degrees_of_freedom_removed=1)
    molecule.prepare_rigid_constraints(algorithm="rattle")

    def deliberately_nontangent(_nfix, _p, _wmass, _dt, _prob, _temperature, *, collective=False):
        assert collective
        return np.array([1.0, 0.0, 0.0, -2.0, 0.0, 0.0])

    monkeypatch.setattr("dynamics.thermostat_driver.thermo_andersen", deliberately_nontangent)
    apply_thermostat(molecule, "andersen", thermo_param=1.0, thermo_temp=300.0, dt=1.0)

    assert molecule._constraint_solver.maximum_velocity_error(
        molecule.q, molecule.p, molecule.wmass
    ) <= molecule._constraint_velocity_tolerance


def test_rigid_restart_validates_constraint_topology(tmp_path):
    def make_rigid(distance):
        molecule = Molecule(
            atoms=["X", "Y"],
            mass=np.array([2.0, 3.0]),
            q_ini=np.array([0.0, 0.0, 0.0, distance, 0.0, 0.0]),
            p_ini=np.zeros(6),
            nfix=1,
        )
        molecule.add_rigid_constraint_group([0, 1], degrees_of_freedom_removed=1)
        molecule.prepare_rigid_constraints()
        return molecule

    original = make_rigid(1.0)
    backfile = tmp_path / "rigid_restart.xyz"
    original._save_restart_state(
        backfile,
        completed_step=8,
        initial_energy=-0.25,
        thermostat=None,
        dt=0.5,
        integrator="verlet",
    )
    state = original._load_restart_state(backfile, completed_step=8)

    matching = make_rigid(1.0)
    assert matching._restore_restart_state(state, None, None, None, 0.5) == -0.25

    mismatched = make_rigid(1.1)
    with pytest.raises(ValueError, match="topology"):
        mismatched._restore_restart_state(state, None, None, None, 0.5)


def test_rigid_constraints_reject_unsupported_integrator():
    molecule = Molecule(
        atoms=["X", "Y"],
        mass=np.ones(2),
        q_ini=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        p_ini=np.zeros(6),
        nfix=1,
    )
    molecule.add_rigid_constraint_group([0, 1], degrees_of_freedom_removed=1)
    molecule.prepare_rigid_constraints()
    molecule.qchem = {"wfu": False}

    with pytest.raises(ValueError, match="supports only"):
        apply_integrator(molecule, "rk4", dt=0.01)
