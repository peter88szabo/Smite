"""Wiring of the FSSH quantum propagator onto Smite's own PES backend.

``fssh.fssh`` is upstream code, unchanged apart from where it gets its
surfaces from. What is exercised here is the glue: that
``fssh.pes_adapter`` serves ``pesrun`` through ``gp_pes``'s shape contract,
that ``dynamics.quantum_driver`` validates and drives it, and that a
trajectory without ``q_integrator`` behaves exactly as it did before.
"""

import numpy as np
import pytest

from core.molecule import Molecule
from utils.atomic_masses import get_mass_vector
from dynamics.quantum_driver import (
    apply_quantum_integrator,
    initialize_quantum_propagator,
)
from fssh import pes_adapter


# A two-surface analytic PES with an avoided crossing along q[0]. ``config["state"]``
# picks the surface, so a single directory serves both electronic states through
# two different qcinput dicts -- the same shape of configuration a real pair such as
# peslib/HO2_3Sigma_negative + peslib/HO2_1Deltag would use across two directories.
PES_INTERFACE = '''
import numpy as np


class PESCalculator:
    K = 0.05

    def __init__(self, pes_dir, config=None):
        config = config or {}
        self.state = int(config.get("state", 0))
        # Widely separated surfaces by default; shrink these to reach the
        # near-degenerate regime where FSSH actually computes couplings.
        self.A = float(config.get("gap_slope", 0.6))
        self.GAP = float(config.get("min_gap", 0.02))

    def _sign(self):
        return -1.0 if self.state == 0 else 1.0

    def _gap(self, s):
        return np.sqrt(self.GAP ** 2 + (self.A * s) ** 2)

    def _dgap(self, s):
        return self.A ** 2 * s / np.sqrt(self.GAP ** 2 + (self.A * s) ** 2)

    def energy(self, q, atoms):
        q = np.asarray(q, dtype=float).reshape(-1)
        return 0.5 * self.K * float(q @ q) + 0.5 * self._sign() * self._gap(q[0])

    def gradient(self, q, atoms):
        q = np.asarray(q, dtype=float).reshape(-1)
        grad = self.K * q.copy()
        grad[0] += 0.5 * self._sign() * self._dgap(q[0])
        return grad

    def force(self, q, atoms):
        return -self.gradient(q, atoms)
'''


ATOMS = ["H", "O", "O"]


@pytest.fixture
def pes_dir(tmp_path):
    directory = tmp_path / "TwoState"
    directory.mkdir()
    (directory / "pes_interface.py").write_text(PES_INTERFACE)
    return directory


@pytest.fixture
def state_qcinput(pes_dir):
    return [
        {"qchem": "PES", "pes_path": str(pes_dir), "state": 0},
        {"qchem": "PES", "pes_path": str(pes_dir), "state": 1},
    ]


@pytest.fixture(autouse=True)
def clear_adapter():
    pes_adapter.clear_states()
    yield
    pes_adapter.clear_states()


@pytest.fixture
def geometry():
    return np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0])


# --------------------------------------------------------------------------
# pes_adapter
# --------------------------------------------------------------------------

def test_adapter_refuses_to_guess_when_unconfigured(geometry):
    with pytest.raises(RuntimeError, match="No electronic states are configured"):
        pes_adapter.PES_Energy(geometry, 0)


def test_adapter_reproduces_gp_pes_shape_contract(state_qcinput, geometry):
    pes_adapter.configure_states(ATOMS, state_qcinput)
    ndim = geometry.size

    assert isinstance(pes_adapter.PES_Energy(geometry, 0), float)
    assert pes_adapter.PES_Force(geometry, 0).shape == (ndim,)
    assert pes_adapter.PES_Hessian(geometry, 0).shape == (ndim, ndim)

    states = [0, 1]
    assert pes_adapter.PES_Energy(geometry, states).shape == (2,)
    assert pes_adapter.PES_Force(geometry, states).shape == (2, ndim)
    assert pes_adapter.PES_Hessian(geometry, states).shape == (2, ndim, ndim)


def test_adapter_defaults_to_every_configured_state(state_qcinput, geometry):
    pes_adapter.configure_states(ATOMS, state_qcinput)
    assert pes_adapter.PES_Energy(geometry).shape == (2,)

    # A single configured surface collapses to a bare value, as gp_pes does.
    pes_adapter.configure_states(ATOMS, state_qcinput[:1])
    assert isinstance(pes_adapter.PES_Energy(geometry), float)


def test_adapter_maps_state_index_onto_the_configured_order(state_qcinput, geometry):
    pes_adapter.configure_states(ATOMS, state_qcinput)
    lower, upper = pes_adapter.PES_Energy(geometry, [0, 1])

    # State 0 is the lower surface of this model everywhere.
    assert lower < upper
    assert pes_adapter.PES_Energy(geometry, 0) == pytest.approx(lower)
    assert pes_adapter.PES_Energy(geometry, 1) == pytest.approx(upper)

    # Reversing the declared order reverses the indices: the adapter takes the
    # caller's ordering as given and never re-sorts by energy.
    pes_adapter.configure_states(ATOMS, list(reversed(state_qcinput)))
    assert pes_adapter.PES_Energy(geometry, 0) == pytest.approx(upper)


def test_adapter_force_is_minus_the_analytic_gradient(state_qcinput, geometry):
    pes_adapter.configure_states(ATOMS, state_qcinput)
    force = pes_adapter.PES_Force(geometry, 0)

    step = 1.0e-6
    numerical = np.empty_like(geometry)
    for index in range(geometry.size):
        plus, minus = geometry.copy(), geometry.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = -(
            pes_adapter.PES_Energy(plus, 0) - pes_adapter.PES_Energy(minus, 0)
        ) / (2.0 * step)

    assert force == pytest.approx(numerical, abs=1.0e-6)


def test_adapter_rejects_an_unknown_state(state_qcinput, geometry):
    pes_adapter.configure_states(ATOMS, state_qcinput)
    with pytest.raises(IndexError, match="out of range"):
        pes_adapter.PES_Energy(geometry, 5)


def test_adapter_copies_qcinput_so_later_mutation_cannot_swap_surfaces(
    state_qcinput, geometry
):
    pes_adapter.configure_states(ATOMS, state_qcinput)
    before = pes_adapter.PES_Energy(geometry, 0)
    state_qcinput[0]["state"] = 1
    assert pes_adapter.PES_Energy(geometry, 0) == pytest.approx(before)


# --------------------------------------------------------------------------
# dynamics.quantum_driver
# --------------------------------------------------------------------------

def test_driver_is_inert_without_a_quantum_integrator():
    assert initialize_quantum_propagator(None, 1, 0) is None

    molecule = Molecule(ATOMS, np.ones(3), np.zeros(9), np.ones(9))
    momenta = molecule.p.copy()
    apply_quantum_integrator(molecule, None, dt=1.0)
    assert molecule.p == pytest.approx(momenta)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(num_states=1, active_state=0), "greater than 1"),
        (dict(num_states=2, active_state=7), "active_state must satisfy"),
    ],
)
def test_driver_rejects_inconsistent_state_settings(state_qcinput, kwargs, message):
    with pytest.raises(ValueError, match=message):
        initialize_quantum_propagator(
            "fssh", atoms=ATOMS, state_qcinput=state_qcinput, **kwargs
        )


def test_driver_requires_one_surface_per_state(state_qcinput):
    with pytest.raises(ValueError, match="lists 2 surface"):
        initialize_quantum_propagator(
            "fssh", 3, 0, atoms=ATOMS, state_qcinput=state_qcinput
        )

    with pytest.raises(ValueError, match="one qcinput per electronic state"):
        initialize_quantum_propagator("fssh", 2, 0, atoms=ATOMS, state_qcinput=None)


def test_driver_rejects_an_unknown_quantum_integrator(state_qcinput):
    with pytest.raises(ValueError, match="You can choose from: fssh"):
        initialize_quantum_propagator(
            "ehrenfest", 2, 0, atoms=ATOMS, state_qcinput=state_qcinput
        )


def test_driver_registers_the_surfaces_it_will_run_on(state_qcinput, geometry):
    propagator = initialize_quantum_propagator(
        "fssh", 2, 0, atoms=ATOMS, state_qcinput=state_qcinput
    )
    assert propagator is not None
    assert pes_adapter.get_states().num_states == 2
    assert pes_adapter.PES_Energy(geometry).shape == (2,)


def test_a_hop_redirects_the_classical_force_to_the_new_surface(state_qcinput):
    """The nuclei must carry on along the state FSSH hopped to, not the old one."""
    initialize_quantum_propagator(
        "fssh", 2, 0, atoms=ATOMS, state_qcinput=state_qcinput
    )

    molecule = Molecule(ATOMS, np.ones(3), np.zeros(9), np.ones(9))
    molecule.qchem = {"qchem": "PES", "pes_path": state_qcinput[0]["pes_path"],
                      "state": 0, "wfu": False}
    molecule.num_states = 2
    molecule.active_state = 0

    def hop_to_state_one(dt, active_state, num_states, q, p, mass, dtq=None):
        return p, 1

    apply_quantum_integrator(molecule, "fssh", dt=1.0, propagator=hop_to_state_one)

    assert molecule.active_state == 1
    assert molecule.qchem["state"] == 1          # surface followed the hop
    assert molecule.qchem["wfu"] is False        # unrelated keys preserved


def test_no_hop_leaves_the_classical_surface_alone(state_qcinput):
    initialize_quantum_propagator(
        "fssh", 2, 0, atoms=ATOMS, state_qcinput=state_qcinput
    )

    molecule = Molecule(ATOMS, np.ones(3), np.zeros(9), np.ones(9))
    qchem = {"qchem": "PES", "pes_path": state_qcinput[0]["pes_path"], "state": 0}
    molecule.qchem = qchem
    molecule.num_states = 2
    molecule.active_state = 0

    apply_quantum_integrator(
        molecule, "fssh", dt=1.0,
        propagator=lambda dt, s, n, q, p, m, dtq=None: (p, s),
    )

    assert molecule.qchem is qchem


def test_dtq_is_given_in_femtoseconds_like_timestep(state_qcinput):
    """``run_trajectory`` hands the driver a ``dt`` already in atomic units.

    ``dtq`` still comes from the caller in femtoseconds, the same unit as
    ``timestep``, so it must be converted before reaching FSSH -- otherwise the
    two timesteps are silently 41.34x out of step with each other.
    """
    from utils.constants import FS_TO_AU_TIME

    initialize_quantum_propagator(
        "fssh", 2, 0, atoms=ATOMS, state_qcinput=state_qcinput
    )

    molecule = Molecule(ATOMS, np.ones(3), np.zeros(9), np.ones(9))
    molecule.qchem = dict(state_qcinput[0])
    molecule.num_states, molecule.active_state = 2, 0

    seen = {}

    def record(dtc, active_state, num_states, q, p, mass, dtq=None):
        seen["dtc"], seen["dtq"] = dtc, dtq
        return p, active_state

    dt_au = 0.25 * FS_TO_AU_TIME
    apply_quantum_integrator(molecule, "fssh", dt_au, dtq=0.025, propagator=record)

    assert seen["dtc"] == pytest.approx(dt_au)
    assert seen["dtq"] == pytest.approx(0.025 * FS_TO_AU_TIME)
    # ten quantum steps per classical step, as the ratio of the inputs implies
    assert int(seen["dtc"] / seen["dtq"]) == 10


def test_dtq_left_unset_keeps_fssh_s_own_default(state_qcinput):
    initialize_quantum_propagator(
        "fssh", 2, 0, atoms=ATOMS, state_qcinput=state_qcinput
    )
    molecule = Molecule(ATOMS, np.ones(3), np.zeros(9), np.ones(9))
    molecule.qchem = dict(state_qcinput[0])
    molecule.num_states, molecule.active_state = 2, 0

    seen = {}

    def record(dtc, active_state, num_states, q, p, mass, dtq=None):
        seen["dtq"] = dtq
        return p, active_state

    apply_quantum_integrator(molecule, "fssh", 10.0, dtq=None, propagator=record)
    assert seen["dtq"] is None


def test_driver_refuses_to_step_without_a_propagator():
    molecule = Molecule(ATOMS, np.ones(3), np.zeros(9), np.ones(9))
    with pytest.raises(ValueError, match="No quantum propagator was initialized"):
        apply_quantum_integrator(molecule, "fssh", dt=1.0, propagator=None)


# --------------------------------------------------------------------------
# End to end, through Molecule.run_trajectory
# --------------------------------------------------------------------------

def _hydrogen_oxygen(pes_dir, tmp_path, coupled):
    """A three-atom system heading into the avoided crossing at q[0] == 0."""
    narrow = {"gap_slope": 0.01, "min_gap": 0.0002} if coupled else {}
    states = [
        {"qchem": "PES", "pes_path": str(pes_dir), "state": index,
         "wfu": False, **narrow}
        for index in (0, 1)
    ]

    mass = get_mass_vector(ATOMS)
    q = np.array([-1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3, 0.0, 0.0])
    p = np.zeros(9)
    p[0] = 0.02 * mass[0]

    molecule = Molecule(ATOMS, mass, q, p)
    molecule.fname = "fssh_wiring"
    molecule.qchem = dict(states[0])

    kwargs = dict(
        integrator="verlet", timestep=0.5, maxstep=120, iprint=1000, Rstop=50.0,
        traj_file=str(tmp_path / "traj.xyz"), backfile=str(tmp_path / "back.xyz"),
    )
    return molecule, states, kwargs


def test_trajectory_without_q_integrator_stays_on_a_single_surface(pes_dir, tmp_path):
    molecule, _, kwargs = _hydrogen_oxygen(pes_dir, tmp_path, coupled=False)
    molecule.run_trajectory(**kwargs)

    # The default still records which surface the nuclei were on ...
    assert molecule.active_state == 0
    assert molecule.num_states == 1

    # ... but no surfaces were ever registered, so the quantum path -- and with
    # it the SciPy import that FSSH needs -- was never entered.
    with pytest.raises(RuntimeError, match="No electronic states are configured"):
        pes_adapter.get_states()


def test_fssh_trajectory_conserves_amplitude_norm_and_hops(pes_dir, tmp_path):
    # Hops are drawn from the global NumPy stream, so the seed is part of the
    # test: without it this assertion is a coin toss.
    np.random.seed(1)
    molecule, states, kwargs = _hydrogen_oxygen(pes_dir, tmp_path, coupled=True)

    visited = []
    norms = []
    propagators = []

    import dynamics.quantum_driver as driver
    import core.molecule as molecule_module

    original = driver.apply_quantum_integrator

    def record(system, name, dt, dtq=None, propagator=None):
        original(system, name, dt, dtq, propagator)
        if propagator is not None:
            propagators.append(propagator)
            visited.append(system.active_state)
            norms.append(float(np.sum(np.abs(propagator.c) ** 2)))

    molecule_module.apply_quantum_integrator = record
    try:
        molecule.run_trajectory(
            q_integrator="fssh", num_states=2, active_state=0,
            state_qcinput=states, dtq=0.05, de_cutoff=0.5, **kwargs
        )
    finally:
        molecule_module.apply_quantum_integrator = original

    assert len(visited) == 120
    assert np.all(np.isfinite(molecule.q)) and np.all(np.isfinite(molecule.p))

    # The electronic amplitudes are propagated unitarily.
    assert norms == pytest.approx([1.0] * len(norms), abs=1.0e-6)

    # Population actually moves onto the upper surface ...
    assert np.abs(propagators[-1].c[1]) ** 2 > 1.0e-6
    # ... and the trajectory hops rather than sitting on one state.
    assert set(visited) == {0, 1}

    # Whatever surface it ends on, the classical force follows it.
    assert molecule.qchem["state"] == molecule.active_state
