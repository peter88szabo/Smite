import math

import numpy as np
import pytest

from core.collision import surface_impact_geometry


@pytest.mark.parametrize("theta", [0.0, math.radians(35.0), math.radians(70.0)])
def test_oblique_surface_geometry_preserves_separation_and_impact_parameter(theta):
    Rini = 18.0
    bimp = 5.0
    displacement, approach = surface_impact_geometry(
        Rini,
        bimp,
        theta,
        normal_direction=np.array([-1.0, 0.0, 0.0]),
        in_plane_direction=np.array([0.0, 0.6, 0.8]),
    )

    np.testing.assert_allclose(np.linalg.norm(displacement), Rini, atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.norm(approach), 1.0, atol=1.0e-12)
    perpendicular = displacement - np.dot(displacement, approach) * approach
    np.testing.assert_allclose(np.linalg.norm(perpendicular), bimp, atol=1.0e-12)


def test_surface_geometry_rejects_an_impact_parameter_larger_than_separation():
    with pytest.raises(ValueError, match="impact parameter"):
        surface_impact_geometry(3.0, 3.1, 0.2, [1, 0, 0], [0, 1, 0])
