from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.constants import ANGSTROM_TO_BOHR


_COVALENT_RADII_ANGSTROM = {
    "H": 0.31, "D": 0.31, "T": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04,
    "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92,
    "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62,
    "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46,
    "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
}


@dataclass(frozen=True)
class InternalCoords:
    nat: int
    bonds: list[tuple[int, int]]
    angles: list[tuple[int, int, int]]
    linear_bends: list[tuple[int, int, int, int]]
    dihedrals: list[tuple[int, int, int, int]]
    impropers: list[tuple[int, int, int, int]]

    @property
    def nint(self) -> int:
        return (
            len(self.bonds)
            + len(self.angles)
            + len(self.linear_bends)
            + len(self.dihedrals)
            + len(self.impropers)
        )

    @property
    def nbonds(self) -> int:
        return len(self.bonds)

    @property
    def nangles(self) -> int:
        return len(self.angles)

    @property
    def nlinear_bends(self) -> int:
        return len(self.linear_bends)

    @property
    def ndihedrals(self) -> int:
        return len(self.dihedrals)

    @property
    def nimpropers(self) -> int:
        return len(self.impropers)

    @property
    def ntorsions(self) -> int:
        return len(self.dihedrals) + len(self.impropers)


@dataclass(frozen=True)
class ConnectivityModel:
    rcov_disp: np.ndarray
    kcn: float = 16.0
    facmin: float = 0.45
    connect_fragments: bool = True


@dataclass(frozen=True)
class BPGMatrix:
    b: np.ndarray
    inv_g: np.ndarray


def default_connectivity_model(atoms, *, kcn=16.0, facmin=0.45, connect_fragments=True):
    radii = [
        _COVALENT_RADII_ANGSTROM.get(str(atom), 0.75) * ANGSTROM_TO_BOHR
        for atom in atoms
    ]
    return ConnectivityModel(
        rcov_disp=np.asarray(radii, dtype=float),
        kcn=float(kcn),
        facmin=float(facmin),
        connect_fragments=bool(connect_fragments),
    )


def _atom(x, i):
    j = 3 * i
    return x[j:j + 3]


def _norm(v):
    return float(np.linalg.norm(v))


def _distance_table(x, nat):
    xyz = np.asarray(x, dtype=float).reshape(nat, 3)
    diff = xyz[:, None, :] - xyz[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _components(con):
    nat = con.shape[0]
    seen = np.zeros(nat, dtype=bool)
    comps = []
    for start in range(nat):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in np.nonzero(con[i])[0]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        comps.append(comp)
    return comps


def _add_fragment_links(con, bonds, rab):
    comps = _components(con)
    while len(comps) > 1:
        best = None
        for ia, comp_a in enumerate(comps[:-1]):
            for ib, comp_b in enumerate(comps[ia + 1:], start=ia + 1):
                for a in comp_a:
                    for b in comp_b:
                        r = rab[a, b]
                        if r <= 0.0:
                            continue
                        if best is None or r < best[0]:
                            best = (r, ia, ib, a, b)
        if best is None:
            break
        _, _ia, _ib, a, b = best
        i, j = (a, b) if a < b else (b, a)
        con[i, j] = True
        con[j, i] = True
        bonds.append((i, j))
        comps = _components(con)


def analyze_structure(x, atoms=None, model=None, *, linear_bend_threshold_degrees=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size % 3 != 0:
        raise ValueError("Internal coordinate analysis requires 3N Cartesian coordinates")
    nat = x.size // 3
    if model is None:
        if atoms is None:
            raise ValueError("atoms or model must be provided for internal coordinate analysis")
        model = default_connectivity_model(atoms)
    if len(model.rcov_disp) != nat:
        raise ValueError(f"Connectivity model has {len(model.rcov_disp)} atoms, geometry has {nat}")

    rab = _distance_table(x, nat)
    con = np.zeros((nat, nat), dtype=bool)
    bonds = []
    for i in range(nat - 1):
        for j in range(i + 1, nat):
            rij = rab[i, j]
            if rij <= 0.0:
                continue
            rco = model.rcov_disp[i] + model.rcov_disp[j]
            fac = 1.0 / (1.0 + np.exp(-model.kcn * (rco / rij - 1.0)))
            if fac > model.facmin:
                con[i, j] = True
                con[j, i] = True
                bonds.append((i, j))

    if model.connect_fragments and nat > 1:
        _add_fragment_links(con, bonds, rab)

    return internal_coords_from_bonds(
        nat,
        bonds,
        x=x,
        linear_bend_threshold_degrees=linear_bend_threshold_degrees,
    )


def internal_coords_from_bonds(nat, bonds_in, x=None, *, linear_bend_threshold_degrees=None):
    con = np.zeros((nat, nat), dtype=bool)
    bonds = []
    for i, j in bonds_in:
        i = int(i)
        j = int(j)
        if i < 0 or j < 0 or i >= nat or j >= nat or i == j:
            raise ValueError(f"Invalid internal-coordinate bond ({i}, {j}) for {nat} atoms")
        a, b = (i, j) if i < j else (j, i)
        if not con[a, b]:
            con[a, b] = True
            con[b, a] = True
            bonds.append((a, b))

    angles = []
    linear_bends = []
    x_arr = None if x is None else np.asarray(x, dtype=float).reshape(-1)
    linear_threshold = (
        None
        if linear_bend_threshold_degrees is None
        else np.deg2rad(float(linear_bend_threshold_degrees))
    )
    for i in range(nat):
        neigh = list(np.nonzero(con[i])[0])
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                ia, ic = int(neigh[a]), int(neigh[b])
                is_linear = False
                if x_arr is not None and linear_threshold is not None:
                    theta = _angle(_atom(x_arr, ia) - _atom(x_arr, i), _atom(x_arr, ic) - _atom(x_arr, i))
                    is_linear = theta >= linear_threshold
                if is_linear:
                    linear_bends.append((ia, i, ic, 0))
                    linear_bends.append((ia, i, ic, 1))
                else:
                    angles.append((ia, i, ic))

    dihedrals = []
    for i in range(nat - 1):
        neigh_i = list(np.nonzero(con[i])[0])
        if len(neigh_i) < 2:
            continue
        for j in range(i + 1, nat):
            if not con[i, j]:
                continue
            neigh_j = list(np.nonzero(con[j])[0])
            if len(neigh_j) < 2:
                continue
            for ii in neigh_i:
                if ii == j:
                    continue
                for jj in neigh_j:
                    if jj == i:
                        continue
                    dihedrals.append((int(ii), i, j, int(jj)))

    impropers = []
    for center in range(nat):
        neigh = list(np.nonzero(con[center])[0])
        if len(neigh) != 3:
            continue
        impropers.append((int(neigh[0]), center, int(neigh[1]), int(neigh[2])))

    return InternalCoords(
        nat=nat,
        bonds=bonds,
        angles=angles,
        linear_bends=linear_bends,
        dihedrals=dihedrals,
        impropers=impropers,
    )


def _angle(ba, bc):
    r1 = _norm(ba)
    r2 = _norm(bc)
    if r1 == 0.0 or r2 == 0.0:
        return 0.0
    c = float(np.dot(ba, bc) / (r1 * r2))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _dihedral(a, b, c, d):
    ab = b - a
    bc = c - b
    cd = d - c
    t = np.cross(ab, bc)
    u = np.cross(bc, cd)
    tu = np.cross(t, u)
    rt2 = float(np.dot(t, t))
    ru2 = float(np.dot(u, u))
    vtvu = np.sqrt(rt2 * ru2)
    rbc = _norm(bc)
    if vtvu == 0.0 or rbc == 0.0:
        return 0.0
    cosang = float(np.dot(t, u) / vtvu)
    sinang = float(np.dot(bc, tu) / (rbc * vtvu))
    return float(np.arctan2(sinang, cosang))


def _perpendicular_axes(axis):
    axis = np.asarray(axis, dtype=float)
    norm = _norm(axis)
    if norm == 0.0:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / norm
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, ref))) > 0.8:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - axis * float(np.dot(axis, ref))
    e1_norm = _norm(e1)
    if e1_norm == 0.0:
        e1 = np.array([0.0, 0.0, 1.0])
    else:
        e1 = e1 / e1_norm
    e2 = np.cross(axis, e1)
    e2_norm = _norm(e2)
    if e2_norm == 0.0:
        e2 = np.array([0.0, 0.0, 1.0])
    else:
        e2 = e2 / e2_norm
    return e1, e2


def _linear_bend_component(a, b, c, component):
    ba = a - b
    bc = c - b
    rba = _norm(ba)
    rbc = _norm(bc)
    if rba == 0.0 or rbc == 0.0:
        return 0.0
    u = ba / rba
    v = bc / rbc
    e1, e2 = _perpendicular_axes(u)
    bend = u + v
    return float(np.dot(bend, e1 if int(component) == 0 else e2))


def compute_internals(x, ic):
    x = np.asarray(x, dtype=float).reshape(-1)
    q = np.zeros(ic.nint, dtype=float)
    nb = ic.nbonds
    na = ic.nangles
    nl = ic.nlinear_bends
    nd = ic.ndihedrals

    for i, (a, b) in enumerate(ic.bonds):
        q[i] = _norm(_atom(x, b) - _atom(x, a))
    for i, (a, b, c) in enumerate(ic.angles):
        q[nb + i] = _angle(_atom(x, a) - _atom(x, b), _atom(x, c) - _atom(x, b))
    for i, (a, b, c, component) in enumerate(ic.linear_bends):
        q[nb + na + i] = _linear_bend_component(_atom(x, a), _atom(x, b), _atom(x, c), component)
    for i, (a, b, c, d) in enumerate(ic.dihedrals):
        q[nb + na + nl + i] = _dihedral(_atom(x, a), _atom(x, b), _atom(x, c), _atom(x, d))
    for i, (a, b, c, d) in enumerate(ic.impropers):
        q[nb + na + nl + nd + i] = _dihedral(_atom(x, a), _atom(x, b), _atom(x, c), _atom(x, d))
    return q


def diff_internals(q2, q1, ndihedrals, nimpropers=0):
    dq = np.asarray(q2, dtype=float) - np.asarray(q1, dtype=float)
    ntorsions = int(ndihedrals) + int(nimpropers)
    if ntorsions:
        for k in range(len(dq) - ntorsions, len(dq)):
            if dq[k] > np.pi:
                dq[k] -= 2.0 * np.pi
            elif dq[k] < -np.pi:
                dq[k] += 2.0 * np.pi
    return dq


def bpg_matrix(x, ic, *, pinv_cutoff=1.0e-14):
    x = np.asarray(x, dtype=float).reshape(-1)
    nx = 3 * ic.nat
    if x.size != nx:
        raise ValueError(f"B matrix expected {nx} coordinates, got {x.size}")

    bmat = np.zeros((ic.nint, nx), dtype=float)
    nb = ic.nbonds
    na = ic.nangles
    nl = ic.nlinear_bends
    nd = ic.ndihedrals

    for i, (a, c) in enumerate(ic.bonds):
        ac = _atom(x, c) - _atom(x, a)
        dist = _norm(ac)
        if dist == 0.0:
            continue
        g = ac / dist
        bmat[i, 3 * a:3 * a + 3] = -g
        bmat[i, 3 * c:3 * c + 3] = g

    for ii, (a, b0, c) in enumerate(ic.angles):
        j = nb + ii
        ba = _atom(x, a) - _atom(x, b0)
        bc = _atom(x, c) - _atom(x, b0)
        rba2 = float(np.dot(ba, ba))
        rbc2 = float(np.dot(bc, bc))
        p = np.cross(bc, ba)
        rp = _norm(p)
        if rp == 0.0 or rba2 == 0.0 or rbc2 == 0.0:
            continue
        inc_a = -np.cross(ba, p) / (rba2 * rp)
        inc_c = np.cross(bc, p) / (rbc2 * rp)
        bmat[j, 3 * a:3 * a + 3] = inc_a
        bmat[j, 3 * c:3 * c + 3] = inc_c
        bmat[j, 3 * b0:3 * b0 + 3] = -inc_a - inc_c

    for ii, (a, b0, c, d) in enumerate(ic.dihedrals):
        j = nb + na + nl + ii
        ab = _atom(x, b0) - _atom(x, a)
        ac = _atom(x, c) - _atom(x, a)
        bc = _atom(x, c) - _atom(x, b0)
        bd = _atom(x, d) - _atom(x, b0)
        cd = _atom(x, d) - _atom(x, c)
        t = np.cross(ab, bc)
        u = np.cross(bc, cd)
        rt2 = float(np.dot(t, t))
        ru2 = float(np.dot(u, u))
        rbc = _norm(bc)
        if rt2 == 0.0 or ru2 == 0.0 or rbc == 0.0:
            continue
        inc_t = np.cross(t, bc) / (rt2 * rbc)
        inc_u = -np.cross(u, bc) / (ru2 * rbc)
        v1 = np.cross(inc_t, bc)
        v2 = np.cross(ac, inc_t)
        v3 = np.cross(inc_u, cd)
        v4 = np.cross(inc_t, ab)
        v5 = np.cross(bd, inc_u)
        v6 = np.cross(inc_u, bc)
        bmat[j, 3 * a:3 * a + 3] = v1
        bmat[j, 3 * b0:3 * b0 + 3] = v2 + v3
        bmat[j, 3 * c:3 * c + 3] = v4 + v5
        bmat[j, 3 * d:3 * d + 3] = v6

    def numerical_row(row, atoms_in_coord, value_func, dx=1.0e-5):
        base_atoms = sorted(set(int(atom) for atom in atoms_in_coord))
        for atom in base_atoms:
            for xyz_idx in range(3):
                coord_idx = 3 * atom + xyz_idx
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[coord_idx] += dx
                x_minus[coord_idx] -= dx
                f_plus = value_func(x_plus)
                f_minus = value_func(x_minus)
                delta = f_plus - f_minus
                if delta > np.pi:
                    delta -= 2.0 * np.pi
                elif delta < -np.pi:
                    delta += 2.0 * np.pi
                bmat[row, coord_idx] = delta / (2.0 * dx)

    for ii, (a, b0, c, component) in enumerate(ic.linear_bends):
        j = nb + na + ii
        numerical_row(
            j,
            (a, b0, c),
            lambda xx, aa=a, bb=b0, cc=c, comp=component: _linear_bend_component(
                _atom(xx, aa), _atom(xx, bb), _atom(xx, cc), comp
            ),
        )

    for ii, (a, b0, c, d) in enumerate(ic.impropers):
        j = nb + na + nl + nd + ii
        numerical_row(
            j,
            (a, b0, c, d),
            lambda xx, aa=a, bb=b0, cc=c, dd=d: _dihedral(
                _atom(xx, aa), _atom(xx, bb), _atom(xx, cc), _atom(xx, dd)
            ),
        )

    g = bmat @ bmat.T
    evals, evecs = np.linalg.eigh(g)
    inv_evals = np.where(evals > pinv_cutoff, 1.0 / evals, 0.0)
    inv_g = (evecs * inv_evals) @ evecs.T
    return BPGMatrix(b=bmat, inv_g=inv_g)


def internal_gradient(bpg, grad_x):
    return bpg.inv_g @ (bpg.b @ np.asarray(grad_x, dtype=float))


def dq_to_dx(bpg, dq):
    dq = np.asarray(dq, dtype=float)
    if dq.size != bpg.b.shape[0]:
        raise ValueError(f"dq length mismatch: got {dq.size}, expected {bpg.b.shape[0]}")
    return bpg.b.T @ (bpg.inv_g @ dq)


def best_fit_dq_to_cart(x, qs, dq, ic, bpg, n_iter=5, rms_tol=1.0e-7):
    x = np.asarray(x, dtype=float).reshape(-1)
    tmpx = x + dq_to_dx(bpg, dq)
    for _ in range(int(n_iter)):
        tmpqs = compute_internals(tmpx, ic)
        achieved = diff_internals(tmpqs, qs, ic.ndihedrals, ic.nimpropers)
        missing = diff_internals(dq, achieved, ic.ndihedrals, ic.nimpropers)
        correction = dq_to_dx(bpg, missing)
        tmpx = tmpx + correction
        internal_rms = float(np.linalg.norm(missing) / np.sqrt(max(1, missing.size)))
        cartesian_rms = float(np.linalg.norm(correction) / np.sqrt(max(1, correction.size)))
        if internal_rms <= rms_tol and cartesian_rms <= rms_tol:
            break
    return tmpx


def initial_internal_hessian(ic):
    diag = np.empty(ic.nint, dtype=float)
    diag[:ic.nbonds] = 1.0
    diag[ic.nbonds:ic.nbonds + ic.nangles] = 0.25
    start = ic.nbonds + ic.nangles
    diag[start:start + ic.nlinear_bends] = 0.25
    start += ic.nlinear_bends
    diag[start:start + ic.ndihedrals] = 0.125
    start += ic.ndihedrals
    diag[start:] = 0.125
    return np.diag(diag)


def model_internal_hessian(
    ic,
    q_values=None,
    *,
    bond_min=0.15,
    bond_max=1.00,
    angle_min=0.08,
    angle_max=0.35,
    linear_bend_value=0.20,
    dihedral_value=0.05,
    improper_value=0.08,
):
    """Build the optional simple valence model Hessian used by the optimizer.

    This is a local Smite model Hessian, not a published Lindh, Schlegel, or
    Fischer-Almlof implementation.  It is a diagonal Hessian in primitive
    internal coordinates with empirical positive curvatures:

    - bond coordinates get distance-dependent force constants;
    - stretched bonds are softened smoothly;
    - angle coordinates are softened when adjacent bonds are stretched;
    - linear bends, dihedrals, and impropers get fixed soft curvatures.

    The purpose is only to provide a better initial preconditioner than the old
    constant diagonal Hessian for minimum searches, constrained scans, RDA
    conditional optimizations, and optional model-start TS searches.  It is not
    used for final vibrational analysis or final TS validation, where an exact
    electronic-structure Hessian should still be used.
    """

    q_values = None if q_values is None else np.asarray(q_values, dtype=float).reshape(-1)
    if q_values is not None and q_values.size != ic.nint:
        raise ValueError(f"Model Hessian expected {ic.nint} internal values, got {q_values.size}")

    diag = np.empty(ic.nint, dtype=float)

    def bond_curvature(index):
        if q_values is None:
            return 0.70
        r = max(float(q_values[index]), 1.0e-8)
        # Around ordinary covalent distances this is close to bond_max.  For
        # forming/breaking bonds it decays smoothly but stays positive.
        stretch = max(0.0, r - 2.2)
        return bond_min + (bond_max - bond_min) * np.exp(-0.85 * stretch)

    for idx in range(ic.nbonds):
        diag[idx] = bond_curvature(idx)

    angle_offset = ic.nbonds
    bond_lookup = {tuple(sorted(bond)): idx for idx, bond in enumerate(ic.bonds)}
    for local_idx, (a, b, c) in enumerate(ic.angles):
        adjacent = []
        for pair in (tuple(sorted((a, b))), tuple(sorted((b, c)))):
            if pair in bond_lookup:
                adjacent.append(bond_curvature(bond_lookup[pair]))
        if adjacent:
            scale = min(adjacent)
            diag[angle_offset + local_idx] = angle_min + (angle_max - angle_min) * min(1.0, scale / bond_max)
        else:
            diag[angle_offset + local_idx] = 0.20

    linear_offset = angle_offset + ic.nangles
    diag[linear_offset:linear_offset + ic.nlinear_bends] = linear_bend_value

    dihedral_offset = linear_offset + ic.nlinear_bends
    diag[dihedral_offset:dihedral_offset + ic.ndihedrals] = dihedral_value

    improper_offset = dihedral_offset + ic.ndihedrals
    diag[improper_offset:] = improper_value
    return np.diag(diag)


def initial_internal_hessian_from_model(ic, q_values=None, model="simple"):
    model = str(model).lower()
    if model in {"simple", "constant", "default"}:
        return initial_internal_hessian(ic)
    if model in {"model", "geometry", "distance", "valence"}:
        return model_internal_hessian(ic, q_values=q_values)
    raise ValueError("internal_hessian_model must be 'simple' or 'model'")
