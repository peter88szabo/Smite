import numpy as np
import pytest

from core.molecule import Molecule
from utils.constants import ANGSTROM_TO_BOHR
from utils.distance import ReactionChannelHysteresis


def _coordinates(distance_angstrom):
    return np.array([0.0, 0.0, 0.0, distance_angstrom * ANGSTROM_TO_BOHR, 0.0, 0.0])


def test_reaction_channel_requires_consecutive_matching_frames():
    hysteresis = ReactionChannelHysteresis(required_steps=3)
    conditions = {"dissociation": [((0, 1), "GT", 1.0)]}

    assert hysteresis.update(_coordinates(1.1), conditions) == (False, "default")
    assert hysteresis.update(_coordinates(1.1), conditions) == (False, "default")
    assert hysteresis.update(_coordinates(0.9), conditions) == (False, "default")
    assert hysteresis.candidate_channel is None
    assert hysteresis.consecutive_steps == 0

    assert hysteresis.update(_coordinates(1.1), conditions) == (False, "default")
    assert hysteresis.update(_coordinates(1.1), conditions) == (False, "default")
    assert hysteresis.update(_coordinates(1.1), conditions) == (True, "dissociation")


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "two"])
def test_reaction_persistence_steps_must_be_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        ReactionChannelHysteresis(value)


def test_trajectory_defers_reaction_assignment_until_channel_persists(monkeypatch, tmp_path):
    molecule = Molecule(
        atoms=["H", "H"],
        mass=np.array([1.0, 1.0]),
        q_ini=_coordinates(0.5),
        p_ini=np.zeros(6),
    )
    molecule.fname = "hysteresis_test"
    molecule.qchem = {"wfu": False}
    molecule._set_trajectory_scratch_dir = lambda traj_file, restart=False: None
    molecule.get_energy = lambda file_wf=None: (0.0, 0.0, 0.0)

    propagation_count = {"value": 0}

    def propagate(system, integrator, dt, propagator):
        propagation_count["value"] += 1
        system.q[3] = 2.0 * ANGSTROM_TO_BOHR

    monkeypatch.setattr("core.molecule.initialize_integrator", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.molecule.apply_integrator", propagate)
    monkeypatch.setattr("core.molecule.prepare_wavefunction_directory", lambda *args, **kwargs: tmp_path)

    molecule.run_trajectory(
        maxstep=5,
        iprint=100,
        traj_file=str(tmp_path / "traj.xyz"),
        backfile=str(tmp_path / "backup.xyz"),
        pairs_to_stop={"dissociation": [((0, 1), "GT", 1.0)]},
        reaction_persistence_steps=2,
    )

    assert propagation_count["value"] == 2
    assert molecule.reaction_channel_candidate == "dissociation"
    assert molecule.reaction_channel_persistence == 2
