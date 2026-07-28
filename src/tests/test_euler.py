import numpy as np

import utils.euler as euler


def _record_uniform_calls(monkeypatch):
    calls = []

    def uniform(lower, upper):
        calls.append((lower, upper))
        return 0.0

    monkeypatch.setattr(euler.random, "uniform", uniform)
    return calls


def test_euler_rot_samples_the_full_polar_cosine_range(monkeypatch):
    calls = _record_uniform_calls(monkeypatch)
    euler.euler_rot(np.zeros(3), np.zeros(3))
    assert calls == [(0, 2 * euler.pi), (-1, 1), (0, 2 * euler.pi)]


def test_coordinate_only_euler_rotation_samples_the_full_polar_cosine_range(monkeypatch):
    calls = _record_uniform_calls(monkeypatch)
    euler.euler_rotQ(np.zeros(3))
    assert calls == [(0, 2 * euler.pi), (-1, 1), (0, 2 * euler.pi)]
