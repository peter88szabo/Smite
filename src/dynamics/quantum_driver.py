"""Selection and application of quantum (electronic) propagators.

The counterpart of :mod:`dynamics.integrator_driver`:
:func:`initialize_quantum_propagator` builds the propagator once per trajectory
and :func:`apply_quantum_integrator` advances it by one classical step.  It is
kept separate from the classical driver because the quantum step runs *after*
the nuclear step, acts on the momenta rather than the positions, and may change
which electronic state the nuclei go on propagating on.

Unlike ``integrator_driver``, FSSH is imported inside the function rather than
at module scope.  ``fssh.baeck_an_nac`` needs SciPy, and a module-level import
here would make every trajectory -- including the single-surface ones that never
ask for surface hopping -- fail on a machine without it.
"""

from utils.constants import FS_TO_AU_TIME


QUANTUM_INTEGRATORS = ("fssh",)


def _validate_name(name):
    if name not in QUANTUM_INTEGRATORS:
        raise ValueError(
            "Non existing quantum integrator. You can choose from: "
            + ", ".join(QUANTUM_INTEGRATORS)
        )


def initialize_quantum_propagator(name, num_states, active_state,
                                  atoms=None, state_qcinput=None, de_cutoff=0.5,
                                  de_corr=0.0, n_substeps=None, hop_gap_max=None):
    """Build the quantum propagator and register the surfaces it runs on.

    Returns ``None`` when ``name`` is ``None``, which is what leaves an ordinary
    single-surface trajectory untouched.

    ``de_corr`` is the strength in Hartree of the energy-based decoherence
    correction; 0, the default, leaves plain fewest-switches surface hopping.
    """
    if name is None:
        return None

    _validate_name(name)

    if num_states < 2:
        raise ValueError(
            "Number of states must be greater than 1 for quantum propagation"
        )
    if not 0 <= active_state < num_states:
        raise ValueError(
            f"active_state must satisfy 0 <= active_state < {num_states}, "
            f"got {active_state}"
        )
    if not state_qcinput:
        raise ValueError(
            "Surface hopping needs one qcinput per electronic state; pass "
            "state_qcinput=[{...}, {...}] to run_trajectory()"
        )
    if len(state_qcinput) != num_states:
        raise ValueError(
            f"state_qcinput lists {len(state_qcinput)} surface(s) but "
            f"num_states={num_states}"
        )

    # Imported here, not at module scope -- see the module docstring.
    from fssh.fssh import FSSH
    from fssh.pes_adapter import configure_states

    if de_corr < 0.0:
        raise ValueError("de_corr is a decoherence strength in Hartree and cannot "
                         "be negative; use 0 to disable the correction")

    configure_states(atoms, state_qcinput)
    if n_substeps is not None and int(n_substeps) < 1:
        raise ValueError("n_substeps is a count of electronic sub-steps per "
                         "classical step and must be at least 1")
    if hop_gap_max is not None and hop_gap_max <= 0.0:
        raise ValueError("hop_gap_max is an energy gap in Hartree and must be "
                         "positive; use None to allow hops across any gap")

    return FSSH(num_states, active_state, de_cutoff, de_corr,
                n_substeps=n_substeps, hop_gap_max=hop_gap_max)


def apply_quantum_integrator(molecule, name, dt, dtq=None, propagator=None):
    """Advance the electronic amplitudes one classical step and rescale momenta.

    Updates ``molecule.p`` and ``molecule.active_state`` in place; a hop shows up
    as a change in ``active_state``.

    ``dt`` arrives in atomic units, because ``run_trajectory`` has already
    multiplied the user's ``timestep`` by ``FS_TO_AU_TIME``. ``dtq`` comes
    straight from the caller and is therefore still in femtoseconds, like
    ``timestep``, so it is converted here. Leaving ``dtq`` unset keeps FSSH's own
    default of one tenth of the classical step, which is already consistent.

    A hop also has to redirect the *classical* force, which
    ``integrators.gradient.force_calc`` reads from a single ``molecule.qchem``
    dict and therefore from a single surface.  On a hop the keys that select the
    surface are merged into ``molecule.qchem`` so the nuclei carry on along the
    state they actually hopped to.  Merging rather than replacing keeps whatever
    else the caller put there (``wfu``, scratch paths, ...).
    """
    if name is None:
        return

    _validate_name(name)

    if propagator is None:
        raise ValueError(
            "No quantum propagator was initialized; call "
            "initialize_quantum_propagator() before apply_quantum_integrator()"
        )

    previous_state = molecule.active_state

    molecule.p, molecule.active_state = propagator(
        dt,
        molecule.active_state,
        molecule.num_states,
        molecule.q,
        molecule.p,
        molecule.wmass,
        dtq=None if dtq is None else dtq * FS_TO_AU_TIME,
    )

    if molecule.active_state != previous_state:
        from fssh.pes_adapter import get_states

        surface = get_states().qcinput(molecule.active_state)
        molecule.qchem = {**(molecule.qchem or {}), **surface}

    # Recorded every step so an ensemble can be checked for internal
    # consistency afterwards: the fraction of trajectories on a state should
    # track the average population of that state, and a drift between them is
    # the signature of missing decoherence.
    if hasattr(molecule, "active_state_history"):
        import numpy as np

        molecule.active_state_history.append(int(molecule.active_state))
        molecule.population_history.append(
            np.abs(np.asarray(propagator.c)) ** 2
        )
