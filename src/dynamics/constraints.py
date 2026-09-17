"""Holonomic distance constraints for rigid molecular fragments.

Smite stores Cartesian coordinates as a flat vector and canonical momenta in
the same ordering.  A rigid fragment is represented here by all of its
intramolecular pair distances,

    sigma_ij(q) = |q_i - q_j|^2 - d_ij^2 = 0.

``shake`` applies the standard iterative position correction using the
constraint gradients at the beginning of the drift.  ``rattle`` projects the
momenta onto the tangent space of the constrained manifold at the corrected
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


MAX_PAIRWISE_SHAKE_ATOMS = 32


def _sparse_rigidity_pairs(reference):
    """Return O(N) diagnostic pairs for a group projected as one rigid body."""
    natom = len(reference)
    anchor_a = 0
    distances = np.linalg.norm(reference - reference[anchor_a], axis=1)
    anchor_b = int(np.argmax(distances))
    axis = reference[anchor_b] - reference[anchor_a]
    areas = np.linalg.norm(np.cross(reference - reference[anchor_a], axis), axis=1)

    anchors = [anchor_a, anchor_b]
    area_scale = max(float(np.linalg.norm(axis)) ** 2, 1.0)
    if float(np.max(areas)) > 1.0e-12 * area_scale:
        anchors.append(int(np.argmax(areas)))

    pairs = set()
    for anchor in anchors:
        for atom in range(natom):
            if atom != anchor:
                pairs.add(tuple(sorted((anchor, atom))))
    return sorted(pairs)


class ConstraintConvergenceError(RuntimeError):
    """Raised when SHAKE or RATTLE cannot satisfy its requested tolerance."""


@dataclass(frozen=True)
class DistanceConstraint:
    atom_i: int
    atom_j: int
    distance_squared: float
    group_index: int


class RigidConstraintSolver:
    """Iterative SHAKE/RATTLE solver for one or more independent rigid groups."""

    def __init__(
        self,
        constraints,
        *,
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
        max_iterations=200,
        degrees_of_freedom_removed=0,
        rigid_groups=(),
        singular_group_indices=(),
    ):
        self.constraints = tuple(constraints)
        self.position_tolerance = float(position_tolerance)
        self.velocity_tolerance = float(velocity_tolerance)
        self.max_iterations = int(max_iterations)
        self.degrees_of_freedom_removed = int(degrees_of_freedom_removed)
        self.rigid_groups = tuple(rigid_groups)
        self.singular_group_indices = frozenset(singular_group_indices)

        if not np.isfinite(self.position_tolerance) or self.position_tolerance <= 0.0:
            raise ValueError("SHAKE position tolerance must be positive")
        if not np.isfinite(self.velocity_tolerance) or self.velocity_tolerance <= 0.0:
            raise ValueError("RATTLE velocity tolerance must be positive")
        if self.max_iterations < 1:
            raise ValueError("SHAKE/RATTLE max_iterations must be at least one")
        if not self.constraints:
            raise ValueError("A rigid constraint solver requires at least one distance constraint")

    @classmethod
    def from_rigid_groups(
        cls,
        groups,
        *,
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
        max_iterations=200,
    ):
        constraints = []
        dof_removed = 0
        rigid_groups = []
        singular_group_indices = []

        for group_index, group in enumerate(groups):
            indices = np.asarray(group["atom_indices"], dtype=int)
            reference = np.asarray(group["reference_positions"], dtype=float)
            if reference.shape != (len(indices), 3):
                raise ValueError("Rigid-group reference coordinates must have shape (natom, 3)")
            if len(indices) < 2:
                continue

            use_rigid_projection = len(indices) > MAX_PAIRWISE_SHAKE_ATOMS
            if use_rigid_projection:
                group_pairs = _sparse_rigidity_pairs(reference)
                constraint_jacobian = None
            else:
                group_pairs = list(combinations(range(len(indices)), 2))
                constraint_jacobian = np.zeros((len(group_pairs), 3 * len(indices)))
            for row, (local_i, local_j) in enumerate(group_pairs):
                displacement = reference[local_i] - reference[local_j]
                distance_squared = float(np.dot(displacement, displacement))
                if not np.isfinite(distance_squared) or distance_squared <= 0.0:
                    raise ValueError("Rigid constraints require finite, nonzero reference distances")
                constraints.append(
                    DistanceConstraint(
                        atom_i=int(indices[local_i]),
                        atom_j=int(indices[local_j]),
                        distance_squared=distance_squared,
                        group_index=group_index,
                    )
                )
                if constraint_jacobian is not None:
                    constraint_jacobian[row, 3 * local_i : 3 * local_i + 3] = (
                        2.0 * displacement
                    )
                    constraint_jacobian[row, 3 * local_j : 3 * local_j + 3] = (
                        -2.0 * displacement
                    )

            group_dof = int(group["degrees_of_freedom_removed"])
            if use_rigid_projection:
                jacobian_rank = None
            else:
                singular_values = np.linalg.svd(
                    constraint_jacobian, compute_uv=False
                )
                rank_tolerance = max(
                    float(singular_values[0]) * 1.0e-10,
                    np.finfo(float).eps,
                )
                jacobian_rank = int(np.count_nonzero(singular_values > rank_tolerance))
            if use_rigid_projection or jacobian_rank < group_dof:
                # Linear and planar rigid bodies are singular when represented
                # only by pair-distance gradients.  Large bodies deliberately
                # use the same O(N) mass-metric rigid-body projection to avoid
                # an O(N^2) constraint set at every dynamics step.
                singular_group_indices.append(group_index)

            rigid_groups.append(
                {
                    "atom_indices": np.array(indices, copy=True),
                    "reference_positions": np.array(reference, copy=True),
                    "degrees_of_freedom_removed": group_dof,
                }
            )
            dof_removed += group_dof

        return cls(
            constraints,
            position_tolerance=position_tolerance,
            velocity_tolerance=velocity_tolerance,
            max_iterations=max_iterations,
            degrees_of_freedom_removed=dof_removed,
            rigid_groups=rigid_groups,
            singular_group_indices=singular_group_indices,
        )

    @staticmethod
    def _atom_masses(wmass, natom):
        wmass = np.asarray(wmass, dtype=float)
        if wmass.shape != (3 * natom,):
            raise ValueError("The Cartesian mass vector is incompatible with the coordinates")
        masses = wmass.reshape((natom, 3))
        if not np.allclose(masses, masses[:, :1], rtol=0.0, atol=0.0):
            raise ValueError("Each atom must have one mass repeated over its three Cartesian components")
        atom_masses = masses[:, 0]
        if np.any(~np.isfinite(atom_masses)) or np.any(atom_masses <= 0.0):
            raise ValueError("Constraint masses must be finite and positive")
        return atom_masses

    @staticmethod
    def _as_xyz(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size % 3 != 0:
            raise ValueError(f"{name} must be a flat Cartesian vector")
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity")
        return values.reshape((-1, 3))

    def maximum_position_error(self, q):
        """Return max |r_ij^2-d_ij^2|/d_ij^2."""
        xyz = self._as_xyz(q, "Coordinates")
        maximum = 0.0
        for constraint in self.constraints:
            rij = xyz[constraint.atom_i] - xyz[constraint.atom_j]
            residual = abs(float(np.dot(rij, rij)) - constraint.distance_squared)
            maximum = max(maximum, residual / constraint.distance_squared)
        return maximum

    def maximum_velocity_error(self, q, p, wmass):
        """Return max |r_ij dot v_ij|/d_ij^2 in inverse atomic time."""
        xyz = self._as_xyz(q, "Coordinates")
        momentum = self._as_xyz(p, "Momenta")
        masses = self._atom_masses(wmass, len(xyz))
        maximum = 0.0
        for constraint in self.constraints:
            i = constraint.atom_i
            j = constraint.atom_j
            rij = xyz[i] - xyz[j]
            relative_velocity = momentum[i] / masses[i] - momentum[j] / masses[j]
            residual = abs(float(np.dot(rij, relative_velocity)))
            maximum = max(maximum, residual / constraint.distance_squared)
        return maximum

    def shake(self, q_old, q_trial, p_drift, dt, wmass):
        """Apply the iterative SHAKE position correction.

        The momentum accumulated during the drift is corrected by the
        corresponding constraint impulse, ``m * delta_q / dt``.
        """
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("SHAKE requires a finite, nonzero timestep")

        old_xyz = self._as_xyz(q_old, "Old coordinates")
        trial_xyz = np.array(self._as_xyz(q_trial, "Trial coordinates"), copy=True)
        momentum = np.array(self._as_xyz(p_drift, "Momenta"), copy=True)
        if old_xyz.shape != trial_xyz.shape or momentum.shape != trial_xyz.shape:
            raise ValueError("SHAKE coordinate and momentum shapes are inconsistent")
        masses = self._atom_masses(wmass, len(trial_xyz))

        # At a linear or planar reference structure, a pure distance-constraint
        # Jacobian is rank deficient for out-of-line/out-of-plane deformation.
        # Project those groups directly onto their rigid-motion manifold before
        # applying ordinary pairwise SHAKE to the regular groups.
        for group_index in self.singular_group_indices:
            group = self.rigid_groups[group_index]
            indices = group["atom_indices"]
            reference = group["reference_positions"]
            group_masses = masses[indices]
            total_mass = float(np.sum(group_masses))

            reference_com = np.sum(reference * group_masses[:, None], axis=0) / total_mass
            trial_com = np.sum(trial_xyz[indices] * group_masses[:, None], axis=0) / total_mass
            reference_centered = reference - reference_com
            trial_centered = trial_xyz[indices] - trial_com
            covariance = reference_centered.T @ (group_masses[:, None] * trial_centered)
            left, _singular_values, right_transpose = np.linalg.svd(covariance)
            rotation = right_transpose.T @ left.T
            if np.linalg.det(rotation) < 0.0:
                right_transpose[-1, :] *= -1.0
                rotation = right_transpose.T @ left.T

            projected = reference_centered @ rotation.T + trial_com
            displacement = projected - trial_xyz[indices]
            trial_xyz[indices] = projected
            momentum[indices] += group_masses[:, None] * displacement / dt

        for _iteration in range(self.max_iterations):
            maximum = 0.0
            for constraint in self.constraints:
                if constraint.group_index in self.singular_group_indices:
                    continue
                i = constraint.atom_i
                j = constraint.atom_j
                rij = trial_xyz[i] - trial_xyz[j]
                residual = float(np.dot(rij, rij)) - constraint.distance_squared
                relative_error = abs(residual) / constraint.distance_squared
                maximum = max(maximum, relative_error)
                if relative_error <= self.position_tolerance:
                    continue

                old_rij = old_xyz[i] - old_xyz[j]
                inverse_mass_sum = 1.0 / masses[i] + 1.0 / masses[j]
                denominator = 2.0 * inverse_mass_sum * float(np.dot(rij, old_rij))
                denominator_scale = (
                    2.0
                    * inverse_mass_sum
                    * np.sqrt(constraint.distance_squared)
                    * max(float(np.linalg.norm(rij)), np.finfo(float).tiny)
                )
                if abs(denominator) <= np.finfo(float).eps * denominator_scale:
                    raise ConstraintConvergenceError(
                        f"SHAKE encountered a singular correction for atoms {i} and {j}"
                    )

                multiplier = -residual / denominator
                impulse_direction = multiplier * old_rij
                trial_xyz[i] += impulse_direction / masses[i]
                trial_xyz[j] -= impulse_direction / masses[j]
                momentum[i] += impulse_direction / dt
                momentum[j] -= impulse_direction / dt

            if maximum <= self.position_tolerance:
                final_error = self.maximum_position_error(trial_xyz.ravel())
                if final_error <= self.position_tolerance:
                    return trial_xyz.ravel(), momentum.ravel()

        final_error = self.maximum_position_error(trial_xyz.ravel())
        raise ConstraintConvergenceError(
            "SHAKE did not converge after "
            f"{self.max_iterations} iterations; maximum relative distance error={final_error:.3e}"
        )

    def project_positions(self, q, wmass):
        """Project a starting geometry onto its stored rigid distances."""
        q = np.asarray(q, dtype=float)
        zero_momentum = np.zeros_like(q)
        projected, _ = self.shake(q, q, zero_momentum, 1.0, wmass)
        return projected

    def rattle(self, q, p_trial, wmass):
        """Apply the iterative RATTLE momentum/velocity correction."""
        xyz = self._as_xyz(q, "Coordinates")
        momentum = np.array(self._as_xyz(p_trial, "Momenta"), copy=True)
        if momentum.shape != xyz.shape:
            raise ValueError("RATTLE coordinate and momentum shapes are inconsistent")
        masses = self._atom_masses(wmass, len(xyz))

        # The mass-metric projection v_i = V_com + omega x r_i is the RATTLE
        # tangent-space projection for a fully rigid body.  It remains regular
        # for planar bodies and uses a pseudoinverse for linear bodies.
        for group_index in self.singular_group_indices:
            group = self.rigid_groups[group_index]
            indices = group["atom_indices"]
            group_masses = masses[indices]
            group_momentum = momentum[indices]
            total_mass = float(np.sum(group_masses))
            center = np.sum(xyz[indices] * group_masses[:, None], axis=0) / total_mass
            relative_positions = xyz[indices] - center
            center_velocity = np.sum(group_momentum, axis=0) / total_mass
            angular_momentum = np.sum(
                np.cross(relative_positions, group_momentum), axis=0
            )
            inertia = np.zeros((3, 3))
            for atom_mass, position in zip(group_masses, relative_positions):
                inertia += atom_mass * (
                    np.dot(position, position) * np.eye(3) - np.outer(position, position)
                )
            angular_velocity = np.linalg.pinv(inertia, rcond=1.0e-12) @ angular_momentum
            rigid_velocity = center_velocity + np.cross(angular_velocity, relative_positions)
            momentum[indices] = group_masses[:, None] * rigid_velocity

        for _iteration in range(self.max_iterations):
            maximum = 0.0
            for constraint in self.constraints:
                if constraint.group_index in self.singular_group_indices:
                    continue
                i = constraint.atom_i
                j = constraint.atom_j
                rij = xyz[i] - xyz[j]
                relative_velocity = momentum[i] / masses[i] - momentum[j] / masses[j]
                residual = float(np.dot(rij, relative_velocity))
                relative_error = abs(residual) / constraint.distance_squared
                maximum = max(maximum, relative_error)
                if relative_error <= self.velocity_tolerance:
                    continue

                inverse_mass_sum = 1.0 / masses[i] + 1.0 / masses[j]
                denominator = inverse_mass_sum * float(np.dot(rij, rij))
                if denominator <= np.finfo(float).tiny:
                    raise ConstraintConvergenceError(
                        f"RATTLE encountered a singular correction for atoms {i} and {j}"
                    )

                multiplier = -residual / denominator
                constraint_impulse = multiplier * rij
                momentum[i] += constraint_impulse
                momentum[j] -= constraint_impulse

            if maximum <= self.velocity_tolerance:
                final_error = self.maximum_velocity_error(q, momentum.ravel(), wmass)
                if final_error <= self.velocity_tolerance:
                    return momentum.ravel()

        final_error = self.maximum_velocity_error(q, momentum.ravel(), wmass)
        raise ConstraintConvergenceError(
            "RATTLE did not converge after "
            f"{self.max_iterations} iterations; maximum velocity error={final_error:.3e} au^-1"
        )
