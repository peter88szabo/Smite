import contextlib
import io
import math
import random

import numpy as np

from sampling.polyrotation import angular_momentum, calcI, polyatom_rotation_sampling
from utils.cenmass import cenmass


def _skew_system():
    q = np.array([
        0.2, 1.1, -0.4,
        -1.3, 0.4, 0.8,
        0.7, -1.5, 1.2,
        1.4, 0.3, -1.1,
    ])
    mass = np.array([12.0, 1.0, 16.0, 14.0]) * 1822.888486
    p = np.array([
        0.4, -0.7, 0.2,
        -0.1, 0.6, -0.5,
        0.9, 0.2, -0.3,
        -0.2, 0.1, 0.8,
    ])
    q, p = cenmass(q, p, mass)
    return q, p, mass


def _direct_inertia(q, mass):
    inertia = np.zeros((3, 3))
    for atom_mass, position in zip(mass, q.reshape((-1, 3))):
        inertia += atom_mass * (
            np.dot(position, position) * np.eye(3) - np.outer(position, position)
        )
    return inertia


def test_calcI_returns_physical_lab_frame_inverse():
    q, _p, mass = _skew_system()
    expected = _direct_inertia(q, mass)

    principal_moments, inertia_inverse = calcI(q, mass)

    np.testing.assert_allclose(principal_moments, np.linalg.eigvalsh(expected))
    np.testing.assert_allclose(inertia_inverse, np.linalg.inv(expected))


def test_fixed_J_sampling_realizes_requested_cartesian_angular_momentum():
    q, p, mass = _skew_system()
    jrot = 7
    random.seed(17)

    with contextlib.redirect_stdout(io.StringIO()):
        sampled_p, target_angmom, _principal_moments = polyatom_rotation_sampling(
            {0: ("Q", jrot)}, mass, q.copy(), p.copy()
        )

    actual_angmom = np.asarray(angular_momentum(q, sampled_p))
    np.testing.assert_allclose(actual_angmom, target_angmom, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(
        np.linalg.norm(actual_angmom),
        math.sqrt(jrot * (jrot + 1.0)),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_thermal_sampling_realizes_requested_cartesian_angular_momentum():
    q, p, mass = _skew_system()
    random.seed(91)

    with contextlib.redirect_stdout(io.StringIO()):
        sampled_p, target_angmom, _principal_moments = polyatom_rotation_sampling(
            {0: ("T", 300.0)}, mass, q.copy(), p.copy()
        )

    actual_angmom = np.asarray(angular_momentum(q, sampled_p))
    np.testing.assert_allclose(actual_angmom, target_angmom, rtol=1.0e-10, atol=1.0e-10)
