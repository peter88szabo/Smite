"""Trajectory boundary, repeated-run and hysteresis restart regressions."""

import numpy as np
import pytest

from core.molecule import Molecule
from utils.constants import ANGSTROM_TO_BOHR


@pytest.fixture
def trajectory(monkeypatch, tmp_path):
    molecule = Molecule(["H", "H"], np.ones(2), np.array([0., 0, 0, .5 * ANGSTROM_TO_BOHR, 0, 0]), np.zeros(6))
    molecule.fname = "boundary_test"
    molecule.qchem = {"wfu": False}
    molecule._set_trajectory_scratch_dir = lambda *args, **kwargs: None
    molecule.get_energy = lambda **kwargs: (0., 0., 0.)
    molecule.steps_taken = 0
    molecule.printed_steps = []
    original_print = molecule.print_trajectory
    def print_frame(handle, istep, *args):
        if handle.name.endswith("traj.xyz"):
            molecule.printed_steps.append(istep)
        original_print(handle, istep, *args)
    molecule.print_trajectory = print_frame
    def propagate(system, *_args):
        system.steps_taken += 1
        system.q[3] += .5 * ANGSTROM_TO_BOHR
    monkeypatch.setattr("core.molecule.apply_integrator", propagate)
    monkeypatch.setattr("core.molecule.initialize_integrator", lambda *args: None)
    monkeypatch.setattr("core.molecule.prepare_wavefunction_directory", lambda *args, **kwargs: tmp_path)
    kwargs = dict(timestep=.25, iprint=100, traj_file=str(tmp_path / "traj.xyz"), backfile=str(tmp_path / "backup.xyz"))
    return molecule, kwargs


@pytest.mark.parametrize("mode", ["pairs", "radius"])
def test_reaction_at_last_propagated_state_is_recorded(trajectory, mode):
    molecule, kwargs = trajectory
    stop = {"pairs_to_stop": {"dissociation": [((0, 1), "GT", 1.2)]}} if mode == "pairs" else {"Rstop": 1.2}
    molecule.run_trajectory(maxstep=2, **stop, **kwargs)
    assert molecule.steps_taken == 2
    assert molecule.printed_steps == [0, 2]
    assert molecule.termination_reason == "stop_condition"
    assert molecule.termination_step == 2
    assert molecule.termination_time_fs == .5
    if mode == "pairs":
        assert molecule.termination_channel == "dissociation"


def test_time_limit_records_terminal_state_without_extra_propagation(trajectory):
    molecule, kwargs = trajectory
    molecule.run_trajectory(maxstep=2, Rstop=10, **kwargs)
    assert molecule.steps_taken == 2
    assert molecule.printed_steps == [0, 2]
    assert molecule.termination_reason == "maxstep"
    assert molecule.termination_channel is None
    assert molecule.termination_step == 2


@pytest.mark.parametrize("first_mode", ["pairs", "radius"])
def test_reusing_molecule_can_switch_stopping_mode(trajectory, first_mode):
    molecule, kwargs = trajectory
    pairs = {"pairs_to_stop": {"dissociation": [((0, 1), "GT", .1)]}}
    radius = {"Rstop": .1}
    first, second = (pairs, radius) if first_mode == "pairs" else (radius, pairs)
    molecule.run_trajectory(maxstep=1, **first, **kwargs)
    molecule.run_trajectory(maxstep=1, **second, **kwargs)
    assert molecule.steps_taken == 0
    assert molecule.termination_reason == "stop_condition"
    assert molecule.termination_step == 0
    if first_mode == "pairs":
        assert molecule.pairstop is None
        assert molecule.reaction_channel_candidate is None
        assert molecule.reaction_channel_persistence == 0
    else:
        assert molecule.Rstop is None
        assert molecule.termination_channel == "dissociation"


@pytest.mark.parametrize("split_step", [1, 2])
def test_persistent_reaction_at_boundary_survives_restart(trajectory, split_step):
    molecule, kwargs = trajectory
    stop = dict(pairs_to_stop={"dissociation": [((0, 1), "GT", .9)]}, reaction_persistence_steps=2)
    molecule.run_trajectory(maxstep=split_step, **stop, **kwargs)
    molecule.run_trajectory(maxstep=2, restart=True, **stop, **kwargs)
    assert molecule.steps_taken == 2
    assert molecule.termination_reason == "stop_condition"
    assert molecule.termination_step == 2
    assert molecule.reaction_channel_persistence == 2
