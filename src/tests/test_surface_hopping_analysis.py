"""Ensemble analysis of surface-hopping trajectories.

The quantity these tests pin is internal consistency: the fraction of
trajectories on state k against the ensemble-average population of state k.
Fewest-switches hopping is constructed to make those equal, and a drift between
them is the standard signature of coherences living too long.
"""

import numpy as np
import pytest

from analysis.surface_hopping import (
    hopping_statistics,
    internal_consistency,
    save_internal_consistency,
)


def _consistent_ensemble(n_traj=200, n_steps=50, seed=0):
    """An ensemble that hops exactly as its populations say it should."""
    rng = np.random.default_rng(seed)
    target = np.linspace(0.0, 0.6, n_steps)          # population draining to state 1

    states, populations = [], []
    for _ in range(n_traj):
        draw = rng.random(n_steps)
        trajectory = (draw < target).astype(int)
        states.append(trajectory)
        populations.append(np.stack([1.0 - target, target], axis=1))
    return list(zip(states, populations))


def test_a_consistent_ensemble_shows_small_deviation():
    result = internal_consistency(_consistent_ensemble())

    assert result["n_trajectories"] == 200
    assert result["fraction"].shape == result["population"].shape
    # Sampling noise only, for 200 trajectories.
    assert result["max_deviation"] < 0.12


def test_the_deviation_shrinks_as_the_ensemble_grows():
    small = internal_consistency(_consistent_ensemble(n_traj=25, seed=1))
    large = internal_consistency(_consistent_ensemble(n_traj=2000, seed=1))
    assert large["max_deviation"] < small["max_deviation"]


def test_an_overcoherent_ensemble_is_flagged():
    """Populations say the state is draining; no trajectory ever hops."""
    n_steps = 50
    target = np.linspace(0.0, 0.9, n_steps)
    populations = np.stack([1.0 - target, target], axis=1)
    stuck = [(np.zeros(n_steps, dtype=int), populations) for _ in range(100)]

    result = internal_consistency(stuck)

    assert result["max_deviation"] == pytest.approx(0.9, abs=1.0e-9)
    assert result["fraction"][-1, 1] == 0.0
    assert result["population"][-1, 1] == pytest.approx(0.9)


def test_fractions_sum_to_one_at_every_step():
    result = internal_consistency(_consistent_ensemble())
    assert result["fraction"].sum(axis=1) == pytest.approx(np.ones(50))


def test_hopping_statistics_counts_transitions():
    # One trajectory hops twice, the other never.
    hopper = np.array([0, 0, 1, 1, 0, 0])
    stayer = np.zeros(6, dtype=int)
    populations = np.tile(np.array([0.7, 0.3]), (6, 1))

    stats = hopping_statistics([(hopper, populations), (stayer, populations)])

    assert list(stats["hops_per_trajectory"]) == [2, 0]
    assert stats["total_hops"] == 2
    assert stats["mean_hops"] == pytest.approx(1.0)
    assert stats["occupancy"].sum() == pytest.approx(1.0)


def test_trajectories_of_unequal_length_are_truncated_not_padded():
    short = (np.zeros(10, dtype=int), np.tile([1.0, 0.0], (10, 1)))
    long = (np.zeros(40, dtype=int), np.tile([1.0, 0.0], (40, 1)))

    result = internal_consistency([short, long])
    assert len(result["step"]) == 10


def test_a_trajectory_without_history_is_rejected():
    with pytest.raises(ValueError, match="no surface-hopping history"):
        internal_consistency([(np.array([], dtype=int), np.zeros((0, 2)))])


def test_an_empty_ensemble_is_rejected():
    with pytest.raises(ValueError, match="no trajectories"):
        internal_consistency([])


def test_molecules_are_accepted_directly():
    """The ensemble is normally a list of finished Molecule objects."""
    class Fake:
        def __init__(self, states, populations):
            self.active_state_history = states
            self.population_history = populations

    populations = np.tile([0.6, 0.4], (20, 1))
    ensemble = [Fake(list(np.zeros(20, dtype=int)), list(populations)) for _ in range(5)]

    result = internal_consistency(ensemble)
    assert result["n_trajectories"] == 5
    assert result["fraction"][:, 0] == pytest.approx(np.ones(20))


def test_output_files_are_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = internal_consistency(_consistent_ensemble(n_traj=20, n_steps=15))

    files = save_internal_consistency(result, "demo", dt_fs=0.5, dpi=60)

    data = np.loadtxt(files["consistency_data_file"])
    header = open(files["consistency_data_file"]).readline()
    assert data.shape == (15, 5)                    # time + 2 fractions + 2 populations
    assert "time_fs" in header and "fraction_0" in header and "population_1" in header
    assert (tmp_path / files["consistency_plot_file"]).exists()
