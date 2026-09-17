# Corrections to the surface-hopping implementation in `src/fssh/`

**Date:** 2026-09-17
**Scope:** `src/fssh/baeck_an_nac.py`, `src/fssh/fssh.py`, `src/dynamics/quantum_driver.py`
**Status:** fixed. Covered by 40 tests across `test_baeck_an_nac.py`, `test_fssh_propagation.py`
and `test_fssh_wiring.py`. Full suite **364 passed, 0 failed**.

---

## 1. Summary

The module approximates nonadiabatic couplings from the curvature of the
adiabatic energy gap rather than from wavefunction matrix elements, which is
what lets surface hopping run on a machine-learned surface that has no
wavefunction at all. The **method is sound and appropriate**. The
**implementation contained two errors that produced wrong physics**, plus
several robustness gaps that could corrupt a trajectory silently.

| # | Problem | Severity | Status |
| --- | --- | --- | --- |
| 1 | Element-wise square used where the chain rule requires an outer product | **Wrong couplings** | fixed |
| 2 | Runge-Kutta amplitude propagation is not unitary; norm diverged to NaN | **Wrong populations** | fixed |
| 3 | `svd` discards the sign of the gap curvature | Wrong physics in one regime | fixed |
| 4 | No sign continuity for the coupling between steps | Corrupts coherent phase | fixed |
| 5 | `dtq` was interpreted in atomic units while `timestep` is in femtoseconds | Wrong quantum timestep | fixed |
| 6 | Division by a vanishing gap unguarded | Possible `inf`/`nan` | fixed |
| 7 | Hop probabilities not bounded by one; a hop became certain every sub-step | **Wrong branching** | fixed |
| 8 | No decoherence correction; the amplitudes stayed coherent indefinitely | **Overcoherent branching** | added |
| 9 | `SyntaxWarning` from LaTeX in a non-raw docstring | Cosmetic | fixed |

None of these raise an exception on their own. They produce plausible-looking
hopping statistics that are wrong, which is why each now has a regression test.

Items 1, 3, 4, 6, 7 and 9 are in upstream code. Item 8 was a stub upstream left empty. Item 5 was introduced by the wiring
added when the module was ported onto Smite's `pesrun` backend.

---

## 2. Background: why couplings are approximated here at all

The exact nonadiabatic coupling vector is

```
d_ij(R) = <Psi_i | grad_R H | Psi_j> / (E_j - E_i)
```

which requires electronic wavefunctions. The `peslib/H2O+Kr+` surfaces are
TorchScript Gaussian-process models that return energies only -- there is no
wavefunction to differentiate. The Baeck-An approximation reconstructs the
coupling magnitude from the curvature of the gap,

```
|d_ij| = 1/2 * sqrt( (d2/dx2)(dE) / dE ),      dE = E_j - E_i
```

needing nothing but energies, gradients and Hessians per state. This is an
established method: introduced by Baeck and An (2007) and developed into
production surface hopping as "curvature-driven" coupling by Barbatti and
co-workers. It is the only coupling scheme that *can* run on a machine-learned
surface of this kind.

Its assumptions bound what the corrected code is good for: two states, same
symmetry, near an avoided crossing. It gives a magnitude along the dominant
curvature direction, not a true vector field; it is unreliable at conical
intersections and carries no geometric phase.

---

## 3. Bug 1 -- the gradient term was an element-wise square

### What was wrong

`baeck_an_nac.py` builds the second derivative of the *squared* gap:

```python
d2de2_dx2 = 2 * (de * d2de_dx2 + dde_dx.ravel() ** 2)   # WRONG
```

The chain rule for `dE**2` is

```
d2(dE^2)/dx_a dx_b = 2 * ( dE * d2(dE)/dx_a dx_b  +  d(dE)/dx_a * d(dE)/dx_b )
```

The second term is the **outer product** of the gap gradient with itself, a
`3N x 3N` matrix. `dde_dx.ravel() ** 2` is an element-wise square, a length-`3N`
vector. NumPy broadcasts it across every row of the matrix instead of raising.

### Why it mattered

The next line exists precisely to cancel that term:

```python
nac_dyad = 1/8 * d2de2_dx2 - np.outer(1/2 * dde_dx, 1/2 * dde_dx)
```

With the correct outer product the two gradient contributions cancel exactly and
leave `(1/4) * dE * d2(dE)/dx dy`, the quantity Baeck-An needs. With the
element-wise square they do not cancel, and the matrix handed to the eigensolver
is not even symmetric.

Verified numerically on random input (`dE = 0.7`, 6 coordinates):

| Gradient term | residual vs. `(1/4)*dE*d2(dE)/dxdy` | symmetric? |
| --- | --- | --- |
| `np.outer(g, g)` (correct) | `1.4e-17` | yes |
| `g ** 2` (as written) | `0.124` | **no** |

### The fix

```python
d2de2_dx2 = 2 * (de * d2de_dx2 + np.outer(dde_dx, dde_dx))
```

plus an explicit symmetrization of the dyad before diagonalization, so rounding
cannot upset the symmetric eigensolver.

### Measured effect on the real surfaces

Coupling between the two `H2O+Kr+` states, old algebra vs corrected, along the
Kr-O approach:

| R (Angstrom) | \|d\| old | \|d\| new | ratio | angle between them |
| --- | --- | --- | --- | --- |
| 2.40 | 0.8924 | 0.9493 | 0.94 | 4.9 deg |
| 3.00 | 0.6185 | 0.6509 | 0.95 | 0.6 deg |
| 4.00 | 0.2727 | 0.1431 | **1.91** | **67.2 deg** |
| 6.00 | 0.1457 | 0.1075 | **1.36** | **74.9 deg** |

Near the well the gap gradient is small compared with `dE * Hessian`, so the
error is a few percent. In the long-range region it dominates, and the old
coupling pointed 67 to 75 degrees away from the correct direction. That is where
a scattering trajectory spends most of its time.

---

## 4. Bug 2 -- the amplitude propagator was not unitary

### What was wrong

The electronic amplitudes were advanced with fourth-order Runge-Kutta:

```python
k1 = _cdot(...); k2 = ...; k3 = ...; k4 = ...
c = c + dtq / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
```

The adiabatic equation of motion is `cdot = A c` with

```
A = -i diag(E) - T ,      T_kj = v . d_kj
```

The couplings are real and antisymmetric, so `T` is anti-Hermitian; `-i diag(E)`
is anti-Hermitian; therefore `A` is anti-Hermitian and the exact propagator
`exp(A dt)` is **unitary**. The electronic norm is a conserved quantity of the
true dynamics.

RK4 is a truncated Taylor expansion of that exponential, so it is not unitary by
construction. The norm drift is not a tuning problem to be minimised -- it is
structural.

### Why it mattered

Hop probabilities are ratios of populations,
`g = 2 dtq Re(...) / |c_a|^2`, so a drifting norm corrupts branching ratios
silently. Measured over 200 quantum steps at a 0.004 Ha gap:

| `v.d` | `dtq` (a.u.) | RK4 \|norm-1\| | unitary \|norm-1\| |
| --- | --- | --- | --- |
| 0.5 | 1.0 | 4.1e-02 | 8.9e-16 |
| 0.5 | 5.0 | 1.0e+00 | 2.1e-14 |
| 2.0 | 1.0 | 1.0e+00 | 2.7e-14 |
| 2.0 | 5.0 | **NaN** | 2.8e-13 |
| 8.0 | 1.0 | **NaN** | 3.5e-14 |
| 8.0 | 5.0 | **NaN** | 5.5e-13 |

RK4 lost the entire norm or diverged to NaN in the strong-coupling regime --
exactly the regime surface hopping exists to describe.

This was partly masked before: because of bug 5 the code was taking 413 quantum
sub-steps per classical step instead of the intended 10, which kept `dtq` small
enough to hide the drift in the test cases.

### The fix

`H = iA` is Hermitian, so the exact propagator is `exp(-i H dtq)`, applied
through a Hermitian eigendecomposition:

```python
hamiltonian = _effective_hamiltonian(v, d, num_states, epot)
eigenvalues, eigenvectors = eigh(hamiltonian)
phases = np.exp(-1.0j * eigenvalues * dtq)
return eigenvectors @ (phases * (eigenvectors.conj().T @ c))
```

with the generator assembled as

```
H_kk = E_k
H_kj = -i (v . d_kj)      for k != j
```

This is exactly unitary for any `dtq`, and on matrices of a handful of states it
is cheaper than four RK4 stages, each of which evaluated `_cdot` in a Python
loop. `_cdot` is retained: it is the literal statement of the equation of motion
and is now used as the reference derivative in the test suite.

---

## 5. Bug 3 -- `svd` discarded the sign of the curvature

```python
U, E, V = svd(nac_dyad)
nac = V[np.argmax(E), :] * np.sqrt(np.max(E)) / de
```

Singular values are non-negative by construction. For a symmetric matrix they
are the absolute values of the eigenvalues, so the sign of the gap curvature was
lost. Where the gap curves *downward* -- no avoided crossing, and Baeck-An
prescribes no coupling -- `sqrt(max(E))` still returned a real, finite coupling
built from a negative eigenvalue's magnitude.

Fixed by diagonalizing with `scipy.linalg.eigh`, which preserves sign, taking
the largest (most positive) eigenvalue, and returning a zero coupling when it is
not positive. `eigh` is also the right tool on its own terms: the dyad is
symmetric, so a general SVD was doing unnecessary work.

---

## 6. Bug 4 -- the coupling sign was not continuous between steps

`self.d` was recomputed from scratch each classical step. The coupling direction
is an eigenvector, whose overall sign is arbitrary -- `eigh` and `svd` may return
either `+v` or `-v` for numerically identical input. The coupling enters the
amplitude derivative as a velocity projection `v . d_ij`, so an arbitrary flip
between consecutive steps flips that term and scrambles the coherent phase.

The previous step's coupling is now threaded through and used to fix the sign:

```python
self.d = _get_couplings(q, epot, num_states, self.de_cutoff, previous=self.d)
```

```python
if previous_nac is not None:
    if np.dot(nac, np.asarray(previous_nac, dtype=float).ravel()) < 0.0:
        nac = -nac
```

Only the sign of `previous_nac` is used. On the first step `self.d` is `None`
and the sign is whatever the eigensolver returns, which is correct since there
is no earlier phase to be continuous with.

---

## 7. Bug 5 -- `dtq` was in the wrong unit

`run_trajectory` converts the user's `timestep` from femtoseconds to atomic
units before propagating (`molecule.py`: `dt = timestep * FS_TO_AU_TIME`). The
quantum driver passed the user's `dtq` straight through, so `dtq` was silently
interpreted in **atomic units** while `timestep` was in **femtoseconds** -- the
two timesteps were out of step by a factor of 41.34.

With `timestep=0.25` and `dtq=0.025`, the intended ten quantum sub-steps per
classical step became 413.

The rest of the codebase already establishes the convention: user-facing times
are in femtoseconds and the driver converts. `dynamics/thermostat_driver.py`
does `tau = thermo_param * FS_TO_AU_TIME` for every thermostat. The quantum
driver now does the same:

```python
dtq=None if dtq is None else dtq * FS_TO_AU_TIME,
```

Leaving `dtq` unset keeps FSSH's internal default of one tenth of the classical
step, which was always self-consistent.

---

## 8. Bug 6 -- unguarded division by the gap

`nac = ... / de` had no protection against `de -> 0`, which is what happens as
two states approach degeneracy. The result would be `inf` or `nan` propagating
into the amplitudes.

A guard now returns a zero coupling below `MINIMUM_GAP = 1e-10` Hartree, where
the two-state model has broken down regardless. The division uses `gap = abs(de)`
so the result does not depend on the order the two states are supplied in.

---

## 9. Bug 7 -- hop probabilities were not bounded by one

The hopping probability

```
g = 2 dtq Re( (v.d_as) conj(c_a) c_s ) / |c_a|^2
```

is first order in ``dtq``, so it is only a probability while it stays below one.
Nothing enforced that. ``_check_hop`` draws a uniform deviate and walks the
cumulative sum, so once the total exceeds one the sum passes the deviate on the
first target state every time: a hop becomes certain every quantum sub-step, and
the branching ratios are destroyed. No error is raised.

The total is now rescaled to one when it would exceed it, which preserves the
relative branching between target states -- the only information still
recoverable once the linearization has failed -- and a `RuntimeWarning` is
emitted once per run:

```python
total = g.sum()
if total > 1.0:
    _warn_hop_probability_exceeded_one()
    g = g / total
```

The warning names the remedy, which is to reduce ``dtq``. Rescaling is a
guard-rail, not a fix for a badly chosen timestep: a run that triggers it should
be repeated with a smaller quantum timestep rather than trusted.

---

## 10. Bug 8 -- no decoherence correction

### What was wrong

`_decoherence()` was an empty stub. Fewest-switches surface hopping propagates
the electronic amplitudes coherently along one classical trajectory, so
coherences that the separation of nuclear wavepackets ought to destroy survive
indefinitely. The populations then drift away from the fraction of trajectories
actually running on each state -- the internal-consistency failure of plain FSSH
-- and the branching ratios drift with them.

### What was added

The energy-based decoherence correction of Granucci and Persico. For every state
`k` other than the active state `a`, in atomic units:

```
tau_k    = (1 + C / E_kin) / |E_k - E_a|
c_k     <- c_k * exp(-dtq / tau_k)
|c_a|^2 <- 1 - sum_{k != a} |c_k|^2
```

so the population of an inactive state decays as `exp(-2 dtq / tau_k)`. Both
factors shorten `tau`: a large energy gap, and fast nuclei, since a large
`E_kin` shrinks the `C / E_kin` term towards the bare `hbar / |E_k - E_a|`. A
motionless system has a diverging `tau` and is left alone. `C` defaults to 0.1
Hartree, the value recommended in the paper.

Applied inside the quantum sub-step loop, after the unitary propagation, so the
coherent step and the damping do not interleave.

### Amplitudes rather than the density matrix

The correction is published as a transformation of `rho`, with four cases: the
inactive populations, the active population, the coherences among inactive
states, and the coherences between the active state and the rest, which pick up
a `sqrt(rho_aa' / rho_aa)` factor.

Smite stores amplitudes, and with `rho_ij = c_i conj(c_j)` the two-line update
above reproduces all four exactly. That is asserted rather than assumed:
`test_the_amplitude_form_matches_the_density_matrix_transformation` writes the
density-matrix form out element by element and checks agreement to 1e-12.

### Off by default

`de_corr` defaults to 0, which leaves plain FSSH and changes no existing result.
`de_corr=0.1` switches the correction on. A negative value is rejected rather
than silently treated as zero.

### Verified

Eleven tests: the analytic `exp(-2 dtq / tau)` decay law; trace conservation;
that the population lost by the inactive states lands exactly on the active one;
the gap and kinetic-energy dependences in the right direction; both divergence
guards; that repeated application drives the active state to one; and the
density-matrix equivalence above.

---

## 11. What was audited and found correct

Not everything suspicious turned out to be a bug. The following were checked in
detail and are right as written:

- **Hop probability.** `g = 2 dtq Re( (v.d_as) conj(c_a) c_s ) / |c_a|^2`. The
  sign convention is easy to get wrong -- it was verified numerically by
  propagating amplitudes and comparing the predicted `g` against the fractional
  population actually leaving the active state. Agreement to a ratio of 1.000.
  Now locked in by `test_hop_probability_tracks_the_population_it_transfers`.
- **Negative-probability clamping** to zero, following Jain, Alguire and
  Subotnik (2016).
- **`_check_hop`.** Cumulative-sum selection against a uniform deviate, with
  `return -1` when no hop is drawn. No fall-through.
- **Velocity rescaling** (`_rescale_velocity`, Hammes-Schiffer & Tully 1994).
  The quadratic `a*gamma^2 - b*gamma - c = 0` with `a = (1/2) sum(d^2/m)`,
  `b = sum(v.d)`, `c = E_a - E_h` is the correct energy-conservation condition,
  and `gamma = (b - sign(b) sqrt(disc)) / (2a)` selects the smaller root, i.e.
  the minimal velocity adjustment.
- **Frustrated-hop reversal.** `gamma = b/a` is exactly energy neutral:
  `dKE = -gamma b + gamma^2 a = 0`.
- **Adiabatic equation of motion** in `_cdot`, including the `1/1j` factor.

---

## 12. Files changed

| File | Change |
| --- | --- |
| `src/fssh/baeck_an_nac.py` | outer product; `eigh` with positive-curvature guard; `previous_nac` for sign continuity; `MINIMUM_GAP` guard; raw docstring |
| `src/fssh/fssh.py` | unitary `exp(-iH dtq)` propagator replacing RK4; new `_effective_hamiltonian`; hop probabilities rescaled to at most one with a warning; energy-based decoherence correction replacing the empty `_decoherence` stub; `_get_couplings` gained `previous` and forwards it; `__call__` passes `self.d` |
| `src/dynamics/quantum_driver.py` | converts `dtq` from femtoseconds to atomic units; exposes `de_corr` |
| `src/tests/test_baeck_an_nac.py` | new, 7 tests |
| `src/tests/test_fssh_propagation.py` | new, 26 tests |
| `src/tests/test_fssh_wiring.py` | 2 tests added for the `dtq` unit convention; the trajectory test seeded and retuned |

The hop selection, frustrated-hop test and velocity rescaling are untouched.

---

## 13. Verification

- Full suite: **364 passed, 0 failed**.
- Electronic norm conserved to `1e-12` or better at couplings and timesteps
  where Runge-Kutta produced NaN.
- FSSH trajectory on `peslib/H2O+Kr+`, 80 steps: norm `1.0000000000`, energy
  conserved to 0.001 kJ/mol, coordinates and momenta finite.
- Model two-state trajectory hops between surfaces in 5 of 5 random seeds, with
  the norm held throughout.
- Single-surface trajectories are unaffected; with `q_integrator=None` none of
  this code runs, and a plain trajectory was verified byte-identical to the
  reference tree.

---

## 14. Known remaining limitations

Not fixed, because they are design choices rather than defects.

1. **Couplings and energies are frozen across the classical step.** Tully's
   formulation interpolates `E` and `v.d` between their values at the start and
   end of the nuclear step. Here the values from the start of the step are held
   for all quantum sub-steps. Adding interpolation needs the couplings evaluated
   after the nuclear move as well, i.e. a restructure of the step.
2. **`de_cutoff` default is effectively inert.** Compared against gaps in
   Hartree but defaulting to `0.5`, which is 13.6 eV. The `H2O+Kr+` gap is about
   0.03 Hartree, so the branch that skips the per-state Hessians never fires.
   Either the default is meant to be eV, or it should be around 0.02 Hartree.
   Changing it changes results, so it was left alone.
3. **Hessians of every state, every step**, wherever the gap is below the cutoff.
   Cheap autograd on the machine-learned surfaces; prohibitive against an ab
   initio backend. This is why the module is wired to a PES backend.
4. **Scope of the approximation.** Two states, same symmetry, avoided crossing.
   The `H2O+Kr+` models are labelled `ci_soc`, so if the states of interest are
   spin-orbit mixed rather than same-symmetry, Baeck-An is outside its stated
   domain and should be checked against a reference method.
5. **Frustrated-hop reversal criterion.** `_check_reverse_velocity` uses the
   active-state force in both of its two conditions. Some formulations of the
   Jasper-Truhlar criterion use the *target*-state force in the second. Worth
   checking against the cited paper; left as written.
---

## 15. References

Cited from memory; worth confirming before quoting in a paper.

- K. K. Baeck, H. An, *J. Chem. Phys.* **126**, 084311 (2007) -- approximation
  of nonadiabatic couplings from the energy gap.
- M. T. do Casal, J. M. Toldo, M. Pinheiro Jr, M. Barbatti, *Open Research
  Europe* (2021) -- fewest-switches surface hopping with Baeck-An couplings.
- J. Westermayr, M. Gastegger, P. Marquetand, *J. Phys. Chem. Lett.* **11**,
  3828 (2020) -- cited in the module docstring.
- S. Hammes-Schiffer, J. C. Tully, *J. Chem. Phys.* **101**, 4657 (1994) --
  velocity rescaling after a hop.
- A. Jain, E. Alguire, J. E. Subotnik, *J. Chem. Theory Comput.* **12**, 5256
  (2016) -- cited for clamping negative hopping probabilities.
- A. W. Jasper, D. G. Truhlar, *Chem. Phys. Lett.* **369**, 60 (2003) --
  frustrated-hop velocity reversal.
- G. Granucci, M. Persico, *J. Chem. Phys.* **126**, 134114 (2007) --
  energy-based decoherence correction.
