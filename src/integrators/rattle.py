"""Constrained velocity-Verlet propagation using SHAKE and RATTLE."""

import numpy as np

from integrators.gradient import force_calc


def constrained_velocity_verlet(
    qcinput,
    dt,
    wmass,
    q,
    p,
    atoms,
    constraint_solver,
    *,
    apply_rattle=True,
):
    """Advance one constrained velocity-Verlet step.

    SHAKE is always used for the position drift.  With ``apply_rattle=True``
    the final momentum is additionally projected with RATTLE.
    """
    q_old = np.asarray(q, dtype=float)
    p_old = np.asarray(p, dtype=float)
    wmass = np.asarray(wmass, dtype=float)

    force = force_calc(qcinput, q_old, atoms)
    p_half = p_old + 0.5 * dt * force
    q_trial = q_old + dt * p_half / wmass

    q_new, p_half = constraint_solver.shake(
        q_old=q_old,
        q_trial=q_trial,
        p_drift=p_half,
        dt=dt,
        wmass=wmass,
    )

    new_force = force_calc(qcinput, q_new, atoms)
    p_new = p_half + 0.5 * dt * new_force
    if apply_rattle:
        p_new = constraint_solver.rattle(q_new, p_new, wmass)

    return q_new, p_new
