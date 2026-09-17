"""Serve FSSH's multi-state PES calls from Smite's own ``qchem_interfaces.pesrun``.

``fssh.py`` was written against ``qchem_interfaces.gp_pes``, which answers for
every electronic state out of a single Gaussian-process model and so needs no
more than ``(q, states)``.  Smite's PES library is organised the other way
round: one surface per directory (``peslib/HO2_1Deltag``,
``peslib/HO2_3Sigma_negative``, ...), each reached by handing ``pesrun`` a
``qcinput`` dict, and each call also wants the atom list.

This module carries that extra context in a module-level registry, set once per
trajectory by :func:`configure_states`, so the call sites inside ``fssh.py``
stay exactly as upstream wrote them.

State ``i`` is whichever surface sits at index ``i`` of the list passed to
:func:`configure_states`.  That order is taken as given and never re-sorted:
FSSH's hop probabilities and energy differences are indexed by it, so silently
reordering the surfaces would change the physics without saying so.

The return shapes mirror ``gp_pes`` exactly, because ``fssh.py`` reshapes them
by hand::

    PES_Energy(q, 1)       -> float
    PES_Energy(q, [0, 1])  -> (n_states,)
    PES_Force(q, 1)        -> (3 * n_atoms,)
    PES_Force(q, [0, 1])   -> (n_states, 3 * n_atoms)
    PES_Hessian(q, 1)      -> (3 * n_atoms, 3 * n_atoms)
    PES_Hessian(q, [0, 1]) -> (n_states, 3 * n_atoms, 3 * n_atoms)

``states=None`` means "every state", collapsing to a bare surface when only one
is configured -- again matching ``gp_pes``.
"""

import numpy as np

from qchem_interfaces.pesrun import PES_Energy as _surface_energy
from qchem_interfaces.pesrun import PES_Force as _surface_force
from qchem_interfaces.pesrun import PES_Hessian as _surface_hessian


_REGISTRY = None


class StatePES:
    """The ordered surfaces backing one multi-state trajectory."""

    def __init__(self, atoms, state_qcinput):
        if not state_qcinput:
            raise ValueError(
                "state_qcinput must name at least one surface, e.g. "
                "[{'qchem': 'PES', 'pes_name': 'HO2_3Sigma_negative'}, "
                "{'qchem': 'PES', 'pes_name': 'HO2_1Deltag'}]"
            )
        self.atoms = list(atoms)
        # Copy the dicts: pesrun caches its calculators on the qcinput contents,
        # so a caller mutating them afterwards would silently swap surfaces.
        self.state_qcinput = [dict(entry) for entry in state_qcinput]

    @property
    def num_states(self):
        return len(self.state_qcinput)

    def qcinput(self, state):
        state = int(state)
        if not 0 <= state < self.num_states:
            raise IndexError(
                f"Electronic state {state} is out of range; "
                f"{self.num_states} surface(s) configured"
            )
        return self.state_qcinput[state]


def configure_states(atoms, state_qcinput):
    """Register the surfaces FSSH may propagate on, lowest index first."""
    global _REGISTRY
    _REGISTRY = StatePES(atoms, state_qcinput)
    return _REGISTRY


def clear_states():
    """Forget the registered surfaces."""
    global _REGISTRY
    _REGISTRY = None


def get_states():
    if _REGISTRY is None:
        raise RuntimeError(
            "No electronic states are configured. Call "
            "fssh.pes_adapter.configure_states(atoms, state_qcinput) before "
            "running a surface-hopping trajectory, or pass state_qcinput to "
            "Molecule.run_trajectory()."
        )
    return _REGISTRY


def _resolve(states, registry):
    """Map ``states`` onto (indices, stack), mirroring ``gp_pes``'s dispatch."""
    if isinstance(states, (int, np.integer)):
        return [int(states)], False
    if states is None:
        if registry.num_states == 1:
            return [0], False
        return list(range(registry.num_states)), True
    return [int(state) for state in states], True


def _coordinates(q):
    return np.asarray(q, dtype=float).reshape(-1)


def PES_Energy(q, states=None):
    registry = get_states()
    indices, stack = _resolve(states, registry)
    q = _coordinates(q)

    energies = np.array(
        [
            float(_surface_energy(None, q, registry.atoms, registry.qcinput(state)))
            for state in indices
        ]
    )
    return energies if stack else float(energies[0])


def PES_Force(q, states=None):
    registry = get_states()
    indices, stack = _resolve(states, registry)
    q = _coordinates(q)

    forces = np.stack(
        [
            np.asarray(
                _surface_force(q, registry.atoms, registry.qcinput(state)), dtype=float
            ).reshape(-1)
            for state in indices
        ]
    )
    return forces if stack else forces[0]


def PES_Hessian(q, states=None):
    registry = get_states()
    indices, stack = _resolve(states, registry)
    q = _coordinates(q)
    ndim = q.shape[0]

    hessians = np.stack(
        [
            np.asarray(
                _surface_hessian(q, registry.atoms, registry.qcinput(state)),
                dtype=float,
            ).reshape(ndim, ndim)
            for state in indices
        ]
    )
    return hessians if stack else hessians[0]
