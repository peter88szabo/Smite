import json

import numpy as np

from normalmode.hessian import _hessian_metadata_path, _hessian_provenance, _load_matching_hessian
from optimizer.minimum import _line_search_cartesian
from optimizer.scan import _backtrack_constrained_trial
from optimizer.transition_state import _ts_trial_is_acceptable, _update_ts_trust_radius
from qchem_interfaces.numerical_hessian import central_difference_hessian
from qchem_interfaces.psi4run import psi4_geometry_specification
from qchem_interfaces.qchem_validation import validate_qchem_input


def test_central_difference_hessian_is_symmetric_and_does_not_mutate_coordinates():
    q = np.array([1.0, -2.0])
    original = q.copy()
    matrix = np.array([[2.0, 0.3], [0.3, 4.0]])

    hessian = central_difference_hessian(q, lambda coordinates: matrix @ coordinates, dx=1.0e-4)

    np.testing.assert_allclose(hessian, matrix, atol=1.0e-10)
    np.testing.assert_array_equal(q, original)


def test_hessian_cache_requires_matching_geometry_and_model(tmp_path):
    qcinput = validate_qchem_input({"qchem": "PES", "pes_name": "test"})
    hessian_file = tmp_path / "hessian.hess"
    hessian = np.eye(3)
    np.savetxt(hessian_file, hessian)
    provenance = _hessian_provenance(qcinput, ["H"], np.zeros(3))
    with open(_hessian_metadata_path(hessian_file), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle)

    loaded, reason = _load_matching_hessian(hessian_file, provenance, (3, 3))
    assert reason is None
    np.testing.assert_allclose(loaded, hessian)

    moved = _hessian_provenance(qcinput, ["H"], np.array([0.0, 0.0, 0.1]))
    loaded, reason = _load_matching_hessian(hessian_file, moved, (3, 3))
    assert loaded is None
    assert "provenance" in reason


def test_failed_minimum_and_scan_line_searches_keep_current_geometry(monkeypatch):
    import optimizer.minimum as minimum

    monkeypatch.setattr(minimum, "_energy", lambda *_args: 1.0)
    monkeypatch.setattr(minimum, "_gradient", lambda *_args: np.array([3.0]))
    q, energy, grad, step, scale, accepted = _line_search_cartesian(
        {}, ["H"], np.array([0.0]), 0.0, np.array([1.0]), np.array([-1.0])
    )
    assert not accepted
    assert scale == 0.0
    np.testing.assert_array_equal(q, [0.0])
    np.testing.assert_array_equal(step, [0.0])
    assert energy == 0.0
    np.testing.assert_array_equal(grad, [1.0])

    q, energy, scale, accepted = _backtrack_constrained_trial(
        np.array([0.0]), 0.0, np.array([1.0]), lambda candidate: candidate, lambda _candidate: 1.0
    )
    assert not accepted
    assert scale == 0.0
    np.testing.assert_array_equal(q, [0.0])
    assert energy == 0.0


def test_ts_acceptance_and_trust_radius_are_safeguarded():
    assert not _ts_trial_is_acceptable(-0.1, np.ones(2), np.ones(2))
    assert not _ts_trial_is_acceptable(np.nan, np.ones(2), np.ones(2))
    assert _ts_trial_is_acceptable(0.5, np.ones(2), np.ones(2))
    assert _update_ts_trust_radius(0.1, np.array([0.1]), 0.9, trust_radius_min=0.01, trust_radius_max=0.3) == 0.2
    assert _update_ts_trust_radius(0.1, np.array([0.01]), 0.1, trust_radius_min=0.01, trust_radius_max=0.3) == 0.05


def test_psi4_geometry_specification_preserves_lab_frame():
    specification = psi4_geometry_specification(0, 2, "H 0.0 0.0 0.0\n")
    assert specification.startswith("0 2\nunits angstrom\n")
    assert "no_com\n" in specification
    assert "no_reorient\n" in specification
