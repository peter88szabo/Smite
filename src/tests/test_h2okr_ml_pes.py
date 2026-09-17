"""The machine-learned H2O + Kr(+) surfaces, reached through pesrun.

The load-bearing assertion here is the unit convention: the TorchScript models
take Cartesian coordinates in **bohr**, which is what Smite's ``q`` already
holds. Going through ``gp_models.GPModel`` instead would multiply them by
``ang2bohr`` and hand the models geometries 1.889x too large -- which does not
raise, it just returns confident nonsense. The GP variance is what catches it.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the H2O+Kr+ surfaces are TorchScript models")

from qchem_interfaces import pesrun
from utils.constants import ANGSTROM_TO_BOHR


PES_DIR = str(pytest.importorskip("pathlib").Path(__file__).resolve().parents[2]
              / "peslib" / "H2O+Kr+")
ATOMS = ["O", "H", "H", "Kr"]
HARTREE_TO_EV = 27.2114


def _geometry(kr_distance_angstrom):
    """O, H, H, Kr with the Kr sitting on the C2 axis, in bohr."""
    return np.array([
        0.0, 0.0, 0.0,
        0.7575, 0.5871, 0.0,
        -0.7575, 0.5871, 0.0,
        0.0, 0.0, kr_distance_angstrom,
    ]) * ANGSTROM_TO_BOHR


def _qcinput(state):
    return {"qchem": "PES", "pes_path": PES_DIR, "state": state}


@pytest.fixture(scope="module", autouse=True)
def models_present():
    import os
    found = [f for f in os.listdir(PES_DIR) if f.endswith(".pt")]
    if len(found) < 2:
        pytest.skip("H2O+Kr+ model weights are not checked out")


def test_both_states_load_and_are_ordered_by_energy():
    q = _geometry(2.6)
    ground = pesrun.PES_Energy(None, q, ATOMS, _qcinput(0))
    excited = pesrun.PES_Energy(None, q, ATOMS, _qcinput(1))
    assert ground < excited


def test_ground_state_reproduces_the_ion_dipole_well():
    """A physical check: a bound well near 2.4 A decaying to zero at long range."""
    energies = {
        R: pesrun.PES_Energy(None, _geometry(R), ATOMS, _qcinput(0)) * HARTREE_TO_EV
        for R in (2.4, 3.0, 7.0)
    }
    assert energies[2.4] < -0.5           # a real well, not a repulsive wall
    assert energies[2.4] < energies[3.0]  # 2.4 A is nearer the minimum than 3.0 A
    assert abs(energies[7.0]) < 0.1       # dissociated


@pytest.mark.parametrize("state", [0, 1])
def test_autograd_force_matches_central_differences(state):
    q = _geometry(2.6)
    qcinput = _qcinput(state)
    force = pesrun.PES_Force(q, ATOMS, qcinput)

    step = 1.0e-5
    numerical = np.empty_like(q)
    for index in range(q.size):
        plus, minus = q.copy(), q.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = -(
            pesrun.PES_Energy(None, plus, ATOMS, qcinput)
            - pesrun.PES_Energy(None, minus, ATOMS, qcinput)
        ) / (2.0 * step)

    assert force == pytest.approx(numerical, abs=1.0e-6)


def test_hessian_is_square_and_symmetric():
    hessian = pesrun.PES_Hessian(_geometry(2.6), ATOMS, _qcinput(0))
    assert hessian.shape == (12, 12)
    assert hessian == pytest.approx(hessian.T, abs=1.0e-9)


def test_results_do_not_depend_on_the_caller_s_atom_order():
    q = _geometry(2.6)
    qcinput = _qcinput(0)

    order = [3, 1, 0, 2]                                  # Kr, H, O, H
    shuffled_atoms = [ATOMS[i] for i in order]
    shuffled_q = q.reshape(4, 3)[order].reshape(-1)

    assert pesrun.PES_Energy(None, shuffled_q, shuffled_atoms, qcinput) == pytest.approx(
        pesrun.PES_Energy(None, q, ATOMS, qcinput)
    )

    force = pesrun.PES_Force(q, ATOMS, qcinput).reshape(4, 3)
    shuffled_force = pesrun.PES_Force(shuffled_q, shuffled_atoms, qcinput).reshape(4, 3)
    assert shuffled_force == pytest.approx(force[order])


def test_coordinates_are_handed_over_in_bohr_not_angstrom():
    """Guard the unit convention using the models' own extrapolation estimate.

    Fed bohr, every geometry along the approach sits inside the training region.
    Fed Angstrom -- i.e. 1.889x too small a molecule -- the variance explodes.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "h2okr_interface", PES_DIR + "/pes_interface.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calculator = module.PESCalculator(PES_DIR, {"state": 0})

    # Fed bohr, the whole approach stays inside the training region.
    for R in (2.4, 3.0, 4.0, 6.0):
        assert calculator.variance(_geometry(R), ATOMS) < 1.0e-2

    # In the interaction region the wrong units are unmistakable. (At long
    # range the shrunken geometry drifts back towards sampled territory, so
    # the check is only diagnostic where the surfaces actually interact.)
    for R in (2.4, 3.0):
        q_bohr = _geometry(R)
        q_angstrom = q_bohr / ANGSTROM_TO_BOHR
        assert (
            calculator.variance(q_angstrom, ATOMS)
            > 100.0 * calculator.variance(q_bohr, ATOMS)
        )
