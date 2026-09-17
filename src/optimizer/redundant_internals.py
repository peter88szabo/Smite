from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizer.internal_coords import (
    BPGMatrix,
    ConnectivityModel,
    InternalCoords,
    analyze_structure,
    bpg_matrix,
    compute_internals,
    default_connectivity_model,
    diff_internals,
    initial_internal_hessian,
    initial_internal_hessian_from_model,
    internal_coords_from_bonds,
)


@dataclass(frozen=True)
class GeneralizedInverse:
    matrix: np.ndarray
    rank: int
    eigenvalues: np.ndarray
    projector: np.ndarray


@dataclass(frozen=True)
class RedundantInternalSystem:
    """Pulay redundant internal coordinate system for one geometry.

    The coordinate list contains primitive valence coordinates.  Redundancy is
    handled algebraically by the generalized inverse of the Wilson G matrix,
    following Pulay and Fogarasi, J. Chem. Phys. 96, 2856 (1992).
    """

    coordinates: InternalCoords
    q: np.ndarray
    b: np.ndarray
    g: np.ndarray
    ginv: np.ndarray
    projector: np.ndarray
    rank: int

    @property
    def nat(self) -> int:
        return self.coordinates.nat

    @property
    def ncart(self) -> int:
        return self.b.shape[1]

    @property
    def nredundant(self) -> int:
        return self.coordinates.nint - self.rank


def generalized_inverse_symmetric(matrix, *, cutoff=1.0e-7):
    """Return the spectral generalized inverse and Pulay projector.

    Pulay recommends replacing ordinary inverses by generalized inverses in
    redundant internal coordinates.  The projector is ``G G-`` and spans the
    nonredundant subspace of the primitive internal coordinates.
    """

    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    evals, evecs = np.linalg.eigh(matrix)
    keep = evals > cutoff
    inv_evals = np.zeros_like(evals)
    inv_evals[keep] = 1.0 / evals[keep]
    ginv = (evecs * inv_evals) @ evecs.T
    projector = matrix @ ginv
    projector = 0.5 * (projector + projector.T)
    return GeneralizedInverse(
        matrix=ginv,
        rank=int(np.count_nonzero(keep)),
        eigenvalues=evals,
        projector=projector,
    )


def build_redundant_internals(
    x,
    atoms=None,
    *,
    model: ConnectivityModel | None = None,
    coordinates: InternalCoords | None = None,
    bonds: list[tuple[int, int]] | None = None,
    linear_bend_threshold_degrees=None,
    g_cutoff=1.0e-7,
):
    """Build primitive redundant internals and their Pulay G inverse."""

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size % 3 != 0:
        raise ValueError("Redundant internal coordinates require 3N Cartesian coordinates")

    if coordinates is not None:
        ic = coordinates
        if ic.nat != x.size // 3:
            raise ValueError(
                f"Internal-coordinate atom count mismatch: got {ic.nat}, expected {x.size // 3}"
            )
    elif bonds is not None:
        ic = internal_coords_from_bonds(
            x.size // 3,
            bonds,
            x=x,
            linear_bend_threshold_degrees=linear_bend_threshold_degrees,
        )
    else:
        if model is None:
            if atoms is None:
                raise ValueError("atoms, model, or bonds must be provided")
            model = default_connectivity_model(atoms)
        ic = analyze_structure(
            x,
            model=model,
            linear_bend_threshold_degrees=linear_bend_threshold_degrees,
        )

    bpg = bpg_matrix(x, ic, pinv_cutoff=g_cutoff)
    g = bpg.b @ bpg.b.T
    gi = generalized_inverse_symmetric(g, cutoff=g_cutoff)
    q = compute_internals(x, ic)
    return RedundantInternalSystem(
        coordinates=ic,
        q=q,
        b=bpg.b,
        g=g,
        ginv=gi.matrix,
        projector=gi.projector,
        rank=gi.rank,
    )


def bpg_from_redundant(system: RedundantInternalSystem):
    """Compatibility adapter for existing optimizer code."""

    return BPGMatrix(b=system.b, inv_g=system.ginv)


def project_internal_vector(system: RedundantInternalSystem, vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size != system.coordinates.nint:
        raise ValueError(
            f"Internal vector length mismatch: got {vector.size}, expected {system.coordinates.nint}"
        )
    return system.projector @ vector


def cartesian_gradient_to_redundant(system: RedundantInternalSystem, grad_x):
    """Pulay Eq. 8 gradient transformation, with force sign handled by caller."""

    grad_x = np.asarray(grad_x, dtype=float).reshape(-1)
    if grad_x.size != system.ncart:
        raise ValueError(f"Cartesian gradient length mismatch: got {grad_x.size}, expected {system.ncart}")
    return system.ginv @ (system.b @ grad_x)


def redundant_step_to_cartesian(system: RedundantInternalSystem, dq):
    """Pulay Eq. 12-13 first-order Cartesian displacement for an internal step."""

    dq = np.asarray(dq, dtype=float).reshape(-1)
    if dq.size != system.coordinates.nint:
        raise ValueError(f"Internal step length mismatch: got {dq.size}, expected {system.coordinates.nint}")
    return system.b.T @ (system.ginv @ dq)


def project_redundant_hessian(system: RedundantInternalSystem, hessian_q, *, redundant_penalty=1000.0):
    """Project a redundant-coordinate Hessian onto the nonredundant subspace.

    ORCA's redundant internal optimizer uses ``P H P + alpha (I - P)`` with
    ``alpha = 1000`` so null/redundant directions have large curvature instead
    of appearing as soft modes in the optimizer Hessian.
    """

    hessian_q = np.asarray(hessian_q, dtype=float)
    nint = system.coordinates.nint
    if hessian_q.shape != (nint, nint):
        raise ValueError(f"Internal Hessian shape mismatch: got {hessian_q.shape}, expected {(nint, nint)}")
    identity = np.eye(nint)
    projected = system.projector @ hessian_q @ system.projector
    if redundant_penalty is not None and float(redundant_penalty) > 0.0:
        projected = projected + float(redundant_penalty) * (identity - system.projector)
    return 0.5 * (projected + projected.T)


def redundant_inverse_hessian(system: RedundantInternalSystem, hessian_q, *, cutoff=1.0e-7):
    """Pulay Eq. 10 inverse Hessian in redundant internal coordinates."""

    projected = project_redundant_hessian(system, hessian_q, redundant_penalty=1000.0)
    inv_projected = generalized_inverse_symmetric(projected, cutoff=cutoff).matrix
    hinv = system.projector @ inv_projected @ system.projector
    return 0.5 * (hinv + hinv.T)


def cartesian_hessian_to_redundant(system: RedundantInternalSystem, hessian_x, *, redundant_penalty=1000.0):
    """Transform a Cartesian Hessian to projected redundant internals.

    This is the linear B-matrix part used in the optimizer.  Curvature terms
    from second derivatives of the internal coordinates are not included here.
    """

    hessian_x = np.asarray(hessian_x, dtype=float)
    if hessian_x.shape != (system.ncart, system.ncart):
        raise ValueError(
            f"Cartesian Hessian shape mismatch: got {hessian_x.shape}, expected {(system.ncart, system.ncart)}"
        )
    hessian_q = system.ginv @ (system.b @ hessian_x @ system.b.T) @ system.ginv
    return project_redundant_hessian(system, hessian_q, redundant_penalty=redundant_penalty)


def initial_redundant_hessian(system: RedundantInternalSystem):
    return project_redundant_hessian(system, initial_internal_hessian(system.coordinates), redundant_penalty=1000.0)


def initial_redundant_hessian_from_model(system: RedundantInternalSystem, model="simple"):
    hessian = initial_internal_hessian_from_model(system.coordinates, q_values=system.q, model=model)
    return project_redundant_hessian(system, hessian, redundant_penalty=1000.0)


def update_cartesian_from_redundant_step(x, system: RedundantInternalSystem, dq, *, n_iter=5):
    """Iteratively solve the nonlinear back-transformation to Cartesians."""

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != system.ncart:
        raise ValueError(f"Cartesian coordinate length mismatch: got {x.size}, expected {system.ncart}")

    target_dq = project_internal_vector(system, dq)
    x_new = x + redundant_step_to_cartesian(system, target_dq)
    for _ in range(int(n_iter)):
        trial = build_redundant_internals(x_new, coordinates=system.coordinates)
        achieved = diff_internals(
            trial.q,
            system.q,
            system.coordinates.ndihedrals,
            system.coordinates.nimpropers,
        )
        missing = diff_internals(
            target_dq,
            achieved,
            system.coordinates.ndihedrals,
            system.coordinates.nimpropers,
        )
        x_new = x_new + redundant_step_to_cartesian(trial, missing)
    return x_new
