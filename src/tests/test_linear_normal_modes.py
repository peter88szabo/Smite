import numpy as np

from normalmode.eckart import eckart_transform, get_eckart_projector
from normalmode.normalmode import getNormalmode


def _linear_triatomic():
    mass = np.array([16.0, 12.0, 16.0])
    q = np.array(
        [
            -2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
        ]
    )
    return mass, q


def test_linear_eckart_projector_has_five_independent_external_modes():
    mass, q = _linear_triatomic()

    projector = get_eckart_projector(mass, q)

    np.testing.assert_allclose(np.trace(projector), 5.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(projector, projector.T, atol=1.0e-12)
    np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-12)


def test_linear_eckart_transform_is_finite():
    mass, q = _linear_triatomic()
    transformed = eckart_transform(mass, q, np.eye(9))
    assert np.all(np.isfinite(transformed))


def test_linear_triatomic_retains_four_vibrational_modes():
    mass, _q = _linear_triatomic()
    hessian = np.diag([0.0] * 5 + [1.0, 2.0, 3.0, 4.0])
    frequencies, external, modes = getNormalmode(
        mass, hessian, linear=True, is_eckart=False
    )

    assert len(external) == 5
    assert len(frequencies) == 4
    assert modes.shape == (9, 4)
