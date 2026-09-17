"""Ensemble analysis of surface-hopping trajectories.

A single surface-hopping trajectory means little on its own; the observable is
the branching over an ensemble. The quantity worth watching is **internal
consistency**: the fraction of trajectories running on state ``k`` should equal
the ensemble-average electronic population of that state,

    N_k(t) / N  ==  < |c_k(t)|^2 >

Fewest-switches hopping enforces this only in the limit of many trajectories and
no spurious coherence. A growing gap between the two curves is the standard
signature that the coherences are living too long -- which is exactly what the
energy-based decoherence correction is there to cure. Comparing a run with and
without ``de_corr`` is the cheapest check that it is doing something.

The histories come from ``molecule.active_state_history`` and
``molecule.population_history``, which ``dynamics.quantum_driver`` fills in on
every quantum step of a surface-hopping run.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _stacked_histories(molecules_or_histories):
    """Accept either a list of molecules or a list of (states, populations)."""
    states, populations = [], []
    for item in molecules_or_histories:
        if hasattr(item, "active_state_history"):
            state_history = item.active_state_history
            population_history = item.population_history
        else:
            state_history, population_history = item

        if len(state_history) == 0:
            raise ValueError(
                "A trajectory carries no surface-hopping history. It was run "
                "without q_integrator, or the history was cleared."
            )
        states.append(np.asarray(state_history, dtype=int))
        populations.append(np.asarray(population_history, dtype=float))

    if not states:
        raise ValueError("no trajectories were supplied")

    shortest = min(len(s) for s in states)
    states = np.vstack([s[:shortest] for s in states])
    populations = np.stack([p[:shortest] for p in populations])
    return states, populations


def internal_consistency(molecules_or_histories, num_states: Optional[int] = None):
    """Compare the fraction of trajectories on each state with its population.

    Parameters
    ----------
    molecules_or_histories
        Either the ``Molecule`` objects of a finished ensemble, or explicit
        ``(active_state_history, population_history)`` pairs.
    num_states
        Taken from the population histories when omitted.

    Returns
    -------
    dict
        ``step`` ``(n_steps,)``; ``fraction`` and ``population``, both
        ``(n_steps, n_states)``; ``deviation``, their difference;
        ``max_deviation``, the largest absolute difference anywhere;
        ``n_trajectories``.
    """
    states, populations = _stacked_histories(molecules_or_histories)
    n_trajectories, n_steps = states.shape
    if num_states is None:
        num_states = populations.shape[2]

    fraction = np.zeros((n_steps, num_states), dtype=float)
    for state in range(num_states):
        fraction[:, state] = np.mean(states == state, axis=0)

    population = np.mean(populations, axis=0)
    deviation = fraction - population

    return {
        "step": np.arange(n_steps),
        "fraction": fraction,
        "population": population,
        "deviation": deviation,
        "max_deviation": float(np.max(np.abs(deviation))),
        "n_trajectories": int(n_trajectories),
    }


def hopping_statistics(molecules_or_histories):
    """Count hops and report how long each state is occupied.

    Returns
    -------
    dict
        ``hops_per_trajectory``; ``total_hops``; ``mean_hops``;
        ``occupancy``, the fraction of all steps spent on each state.
    """
    states, populations = _stacked_histories(molecules_or_histories)
    num_states = populations.shape[2]

    hops = np.sum(states[:, 1:] != states[:, :-1], axis=1)
    occupancy = np.array(
        [float(np.mean(states == state)) for state in range(num_states)]
    )

    return {
        "hops_per_trajectory": hops,
        "total_hops": int(np.sum(hops)),
        "mean_hops": float(np.mean(hops)),
        "occupancy": occupancy,
    }


def save_internal_consistency(consistency, fname: str, *, dt_fs: Optional[float] = None,
                              dpi: int = 600, label: str = ""):
    """Write the internal-consistency comparison as a .dat and a .png."""
    import matplotlib.pyplot as plt

    step = np.asarray(consistency["step"], dtype=float)
    time = step * dt_fs if dt_fs is not None else step
    axis_label = "Time [fs]" if dt_fs is not None else "Quantum step"

    fraction = np.asarray(consistency["fraction"], dtype=float)
    population = np.asarray(consistency["population"], dtype=float)
    num_states = fraction.shape[1]

    rows = [
        (t,) + tuple(fraction[i]) + tuple(population[i])
        for i, t in enumerate(time)
    ]
    header = (
        ("time_fs " if dt_fs is not None else "step ")
        + " ".join(f"fraction_{k}" for k in range(num_states))
        + " "
        + " ".join(f"population_{k}" for k in range(num_states))
    )
    data_file = f"{fname}_surface_hopping_consistency{label}.dat"
    np.savetxt(data_file, np.asarray(rows, dtype=float), fmt='%.8e',
               delimiter=' ', header=header)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for state in range(num_states):
        line, = ax.plot(time, fraction[:, state], lw=2,
                        label=f"state {state}: trajectories")
        ax.plot(time, population[:, state], lw=1.5, ls='--',
                color=line.get_color(), label=f"state {state}: population")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Fraction / population")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"{fname} internal consistency, {consistency['n_trajectories']} trajectories "
        f"(max deviation {consistency['max_deviation']:.3f})"
    )
    ax.legend(ncol=2, fontsize="small")

    plot_file = f"{fname}_surface_hopping_consistency{label}.png"
    fig.savefig(plot_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {"consistency_data_file": data_file, "consistency_plot_file": plot_file}
