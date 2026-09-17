# All-element X-ray scattering tables

**Date:** 2026-09-17
**Scope:** `src/analysis/xray_scattering_tables.py` (new), `src/analysis/data/` (new),
`src/analysis/scattering.py`, `MANIFEST.in`
**Status:** done. 64 tests in `src/tests/test_xray_scattering_tables.py`. Full suite 324 passed.

---

## 1. What this is, and what it is not

`analysis/xray_scattering_tables_hcno.py` covers H, C, N and O and hardcodes
their coefficients. **That is by design, not a defect** -- the filename says so,
and the module was written for the light-element systems it was needed for. It
has not been modified.

What was missing was coverage for everything else, which meant the
`peslib/H2O+Kr+`, `Cl+CH4`, `F+H2` and `ArH2+` systems could not produce
scattering form factors at all:

```
KeyError: 'Unsupported element: Kr'
```

It failed loudly rather than silently, which was the right behaviour.

Equally by design: `get_scattering_form_factors` requires an **NVE** run.
`spectrum=True` only saves geometries when no thermostat is active
(`core/molecule.py`: `if spectrum and thermostat is None`). Scattering off a
thermostatted trajectory would not mean what the IAM formula assumes.

## 2. Source data

The H/C/N/O module names its provenance as the ESRF DABAX library. Tracing it
exactly:

| Quantity | DABAX file | Underlying reference |
| --- | --- | --- |
| Elastic f0(k) | `f0_InterTables.dat` | International Tables Vol. C, in the Cromer-Mann form of Cromer & Mann, *Acta Cryst.* (1968) **A24**, 321 |
| Inelastic S(x) | `isf_Hubbell.dat` | Hubbell, Veigele, Briggs, Brown, Cromer & Howerton, *J. Phys. Chem. Ref. Data* **4**, 471 (1975) |

Downloaded from <https://github.com/oasys-kit/DabaxFiles> and committed verbatim
to `src/analysis/data/` (36 kB and 347 kB). Keeping the original files rather
than a re-encoding means the provenance stays auditable and the parsing can be
re-checked against the published tables.

**One correction to the existing docstring.** The H/C/N/O module states its
elastic data comes from `f0_CromerMann.dat`. It does not: that file's hydrogen
entry is `a1 = 0.38422`, whereas the module has `a1 = 0.4899180`, which is the
`H.` entry of `f0_InterTables.dat`. The two files share the Cromer-Mann
functional form but hold different coefficient sets. The module's own comment
-- *"Neutral free-atom hydrogen entry `H.` from DABAX"* -- is the accurate one.
Left as-is, since that module was not to be touched.

## 3. Neutral-atom entry convention

`f0_InterTables.dat` holds 214 entries: neutral atoms under a plain symbol, ions
as `O1-`, `Pu4+` and so on, plus four dotted entries (`H.`, `C.`, `O2-.`,
`Si.`).

The rule reproduced here is the one the existing module used:

- plain `<El>` for every element, **except**
- hydrogen, which uses `H.`, the neutral free atom rather than the bonded entry.

Verified: C, N and O plain entries match the hardcoded module exactly, and `H.`
matches it digit for digit.

## 4. Coverage

- Elastic table: 98 elements with a neutral plain-symbol entry.
- Inelastic table: 99 elements.
- **Intersection, and therefore what is supported: 98 elements** (only `Es` is
  inelastic-only).

All elements in `peslib/` are covered: H, C, N, O, F, Cl, Ar, Kr.

## 5. Agreement with the existing module

Over `q` in 0 to 25 A^-1, 400 points, the new tables reproduce the hardcoded
module **bit for bit**:

| Quantity | max abs difference |
| --- | --- |
| `f0` for H, C, N, O | `0.000e+00` |
| `isf` for H, C, N, O | `0.000e+00` |
| IAM elastic / inelastic / total, water | `0.000e+00` |

So pointing `analysis/scattering.py` at the wider tables changes no existing
result. That is asserted with `rel=0.0, abs=0.0` in the test suite rather than
merely being observed once.

## 6. Physical checks on the parsed data

`f0(q=0)` is the number of electrons, so it should return Z:

| Element | f0(0) | Z |
| --- | --- | --- |
| H | 1.000 | 1 |
| C | 5.999 | 6 |
| O | 7.999 | 8 |
| Cl | 17.001 | 17 |
| Ar | 17.999 | 18 |
| Kr | 35.996 | 36 |
| Fe | 25.990 | 26 |
| I | 53.001 | 53 |

Across all 98 supported elements the worst deviation is **0.06 electrons**,
which is the accuracy of the four-Gaussian fit.

**With four exceptions, and they are not random.** Extending the same check to
the 113 ionic entries showed that the published actinide block contains two
swapped pairs:

| Entry | Coefficients sum to | Should be | Actually matches |
| --- | --- | --- | --- |
| `Np3+` | 87.000 | 90 | Np6+ |
| `Np6+` | 89.997 | 87 | Np3+ |
| `Np4+` | 93.965 | 89 | neutral Pu |
| `Pu` | 88.999 | 94 | Np4+ |

Uranium immediately above is perfectly consistent (92 / 89 / 88 / 86), and
`Pu3+`, `Pu4+` and `Pu6+` are all correct, so this is localised to those four
entries rather than a systematic actinide problem. The parser reproduces the
file faithfully; this is a defect in the published table.

Rather than silently correcting published data, or hardcoding a blacklist, the
library derives the check from the data itself: `electron_count_error(species)`
returns `f0(0)` minus the electron count, and any species exceeding half an
electron raises a `RuntimeWarning` once. That flags these four today and would
catch any similar defect in a future revision of the table.

Also checked: `f0` decreases monotonically with q; `S(q -> 0) = 0`; unknown
symbols raise `KeyError`; symbols are case-insensitive; scalar and array inputs
agree.

## 7. Charged fragments

The elastic file holds 214 entries: 98 neutral atoms and 113 ionic species
(plus dotted variants). All 113 ions are now exposed and usable anywhere an
element symbol is accepted.

Three spelling quirks in the source had to be absorbed:

- most ions are `O1-`, `Fe3+`;
- one is dotted, `O2-.`, which is the only entry for the oxygen dianion;
- two invert the sign and digit, `Fe+2` and `Ru+4`.

Callers may write any of `O1-`, `O2-`, `O-2`, `Fe2+`, `Fe+2`, in any case; all
resolve to the same canonical `(element, charge)` key.

**The inelastic term for an ion is an approximation.** `isf_Hubbell.dat`
tabulates neutral atoms only -- there are no ionic entries at all. An ion's
incoherent scattering function therefore falls back to its neutral parent, so
for a charged fragment the elastic term is exact and the inelastic term is not:
it corresponds to the neutral atom rather than to a species with a different
electron count. This is announced rather than hidden -- a `RuntimeWarning` once
per ionic species, and `isf_is_approximated(species)` to test for it in code.

Verified: `f0(0)` reproduces `Z - charge` for all 113 ions to better than 0.06
electrons, apart from the three neptunium entries above; and an `O2-`/H/H
fragment scatters more strongly than neutral water in the elastic channel,
with the inelastic channel unchanged by construction.

## 7b. End-to-end

An NVE surface-hopping trajectory on `peslib/H2O+Kr+`, 40 steps with
`spectrum=True`, produced form factors through the normal entry point. The
elastic signal at `q = 0` is 2115.5 against the expected
`(8 + 1 + 1 + 36)^2 = 2116` -- the forward-scattering limit for O/H/H/Kr.

## 8. Fourier transforms of the time-resolved signal

`get_scattering_form_factors` previously wrote only the time-dependent map. Two
transforms of that map are now written as well, each as a `.dat` and a `.png`,
under `fourier=True` (the default):

| Transform | Output | Columns | Answers |
| --- | --- | --- | --- |
| along time | `<fname>_scattering_power_spectrum.*` | `frequency_cm^-1 q_A^-1 elastic inelastic total` | which vibrational frequencies modulate the scattering at each q |
| sine transform along q | `<fname>_scattering_pair_distribution.*` | `time_fs r_A elastic inelastic total` | which interatomic distances are present, and how they move |

Implemented in `analysis/scattering_fourier.py`, reusing
`analysis.spectrum.frequency_axis_cm1` so the frequency axis matches
`molecule.vibrational_spectrum` exactly rather than introducing a second
convention.

**All three channels are transformed** -- elastic, inelastic and total -- and
laid out like the direct output: three columns in one `.dat`, three stacked
panels in one `.png`.

The signal sits in the elastic channel, and that is a property of the model
rather than of the implementation. In the IAM the inelastic term is
`sum_i S_i(q)`, a function of the atom list and q alone. It is not small: on a
400-step GFN2-xTB formaldehyde trajectory it is 72% of the total intensity at
`q = 8` A^-1 and 81% at `q = 12` A^-1. But its spread over the trajectory is
exactly `0.000e+00`, so once the static part is subtracted the inelastic
transform is zero to rounding and the total reproduces the elastic:

| Channel | Power spectrum, max | dPDF, max abs |
| --- | --- | --- |
| elastic | 1.96e5 | 1.96 |
| inelastic | 6.9e-22 | 0 exactly |
| total | equals elastic to 1e-12 .. 1e-8 | equals elastic to 1e-12 |

All three are written so the degeneracy is visible in the output instead of
being asserted in a comment.

**Both transform a difference signal.** The static scattering is orders of
magnitude larger than its modulation, so transforming the raw intensity buries
the dynamics under the time-independent term. `temporal_reference="mean"`
removes the time average, which is what a power spectrum needs; the pair
distribution defaults to `pdf_reference="first"`, the experimental
`dI = I(t) - I(t<0)` convention.

The pair distribution integrates the modified intensity
`sM(q) = q dI(q) / sum_i f_i(q)^2` against `sin(qr)` with a Gaussian damping
defaulting to `ln(10)/qmax^2`, which suppresses the ripple from truncating the
integral at `qmax`. Real-space resolution is about `pi/qmax`.

A Hann window is applied along time by default. A trajectory is short, so
without apodization the truncation sidelobes of the strongest mode are easily
mistaken for weaker modes.

Tests (`src/tests/test_scattering_fourier.py`, 29) check all three against signals
whose answer is known beforehand: a modulation injected at a chosen wavenumber
must return at that wavenumber, and a bond moved from `r0` to `r1` must produce
a depletion lobe at `r0` and a growth lobe at `r1`.

### Short-time transform and offline re-analysis

A single power spectrum assumes stationarity over the whole trajectory, which is
wrong for a reactive run where modes appear, shift and vanish.
`short_time_power_spectrum` slides a window instead, giving `P(t, omega)`.

The existing `analysis/short_time_spectrum.py` could not be reused: its
signature is `(positions, velocities, masses)`, it requires `(samples, 3N)`
histories, it mass-weights and projects out translation and rotation, and its
output is a density in Hartree per cm^-1 normalised to vibrational kinetic
energy. None of that applies to an intensity. The new function is therefore
generic -- it takes any evenly sampled `(n_frames, n_channels)` array and never
interprets the channels -- while reusing the same `window_fs` / `hop_fs`
vocabulary so the two remain comparable.

Verified on a trace whose frequency switches from 1800 to 900 cm^-1 halfway:
with an 80 fs window the spectrogram reports 1668 cm^-1 in the early windows and
834 cm^-1 in the late ones, both the nearest bin at that window's 417 cm^-1
resolution. A companion test confirms the global transform cannot separate the
two regimes at all.

Output, laid out like the others:

```
<fname>_scattering_short_time_spectrum.dat   # time_fs frequency_cm^-1 elastic inelastic total
<fname>_scattering_short_time_spectrum.png   # 3 panels, time versus frequency
```

It is integrated over q, since resolving time, frequency and q simultaneously no
longer fits the two-axis file layout; `short_time_q_range` restricts that
integration to isolate a distance range. A trajectory too short to hold a window
skips the output rather than failing, leaving the other results intact.

`reanalyze_scattering_short_time(path, window_fs=..., hop_fs=..., q_range=...,
label=...)` reads a saved `*_time_dependent_scattering_form_factors.dat` back and
re-transforms it, so a window can be changed without repeating the trajectory;
`label` keeps several analyses of one run side by side. The reader,
`read_time_dependent_scattering`, validates that the file really is a complete
time-by-q grid rather than silently reshaping whatever it is given.

A bug was found and fixed while wiring this up: `Molecule.get_scattering_form_factors`
was a thin wrapper with a fixed argument list, so none of the new options
reached the implementation. It now forwards `**kwargs`.

## 9. Files

| File | Change |
| --- | --- |
| `src/analysis/data/f0_InterTables.dat` | new, verbatim DABAX |
| `src/analysis/data/isf_Hubbell.dat` | new, verbatim DABAX |
| `src/analysis/xray_scattering_tables.py` | new; same API and formulae as the H/C/N/O module, parsing the shipped tables lazily with caching |
| `src/analysis/scattering.py` | one import line repointed at the all-element module |
| `src/tests/test_xray_scattering_tables.py` | new, 64 tests |
| `src/analysis/scattering_fourier.py` | new; the two transforms |
| `src/tests/test_scattering_fourier.py` | new, 29 tests |
| `src/core/molecule.py` | `get_scattering_form_factors` forwards `**kwargs` |
| `MANIFEST.in` | ships `src/analysis/data/*.dat` |
| `README.md` | new "X-ray scattering form factors" section |
| `src/analysis/xray_scattering_tables_hcno.py` | **unchanged** |

## 10. Notes

- The data is parsed on first use and cached, so the 347 kB inelastic file costs
  nothing until scattering is actually requested.
- `analysis/scattering.py` imports `matplotlib` at call time, so the `analysis`
  extra is needed to produce the plot.
- Ionic form factors are exposed; see section 7.
