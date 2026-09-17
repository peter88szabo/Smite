# Acetone: time-resolved X-ray scattering

Self-contained example. Run it from anywhere; every file lands in this
directory.

```bash
python example_scattering_acetone_multimode.py
```

Acetone on GFN2-xTB, 1024 steps of 0.5 fs (512 fs), NVE. The C=O bond and the
two C-C bonds are displaced by different amounts at the start, which puts energy
into several modes at once rather than just one.

## Why acetone

Formaldehyde has essentially a single heavy-atom mode. Acetone has a well
separated set, so the scattering carries more than one frequency, and the
electron-density weighting of X-ray scattering becomes visible.

| Detected | GFN2 harmonic | Assignment | Relative weight |
| --- | --- | --- | --- |
| 846.9 cm^-1 | 827 / 854 | C-C-C skeletal | 1.000 |
| 1759.0 cm^-1 | 1764 | C=O stretch | 0.124 |
| 1368.1 cm^-1 | 1364 | methyl deformation | 0.009 |
| 3062.0 cm^-1 | 3063 / 3072 | C-H stretch | 0.008 |

Every heavy-atom distance swings 0.17 to 0.22 A during the run, so the C-H
stretch is not weak for lack of motion: hydrogen contributes one electron
against six for carbon and eight for oxygen, so it barely scatters. That is the
point worth taking from these plots.

## Output

Each of the following is written as a `.dat` and a `.png`, with elastic,
inelastic and total side by side.

| File | Axes | Shows |
| --- | --- | --- |
| `*_time_dependent_scattering_form_factors` | time, q | `I(q, t)`, the raw signal |
| `*_scattering_power_spectrum` | frequency, q | which frequencies modulate the scattering, and at which q |
| `*_scattering_pair_distribution` | time, r | `dPDF(r, t)`: which interatomic distances move, and when |
| `*_scattering_short_time_spectrum` | time, frequency | `P(t, omega)`, q integrated: when each frequency is active |
| `*_scattering_short_time_spectrum_q_resolved` | time, frequency, q | the full `P(t, omega, q)`; the plot plots frequency against q for several time windows |

Reading the pair distribution plot: the band near r = 1.0 to 1.5 A is the C=O
(1.215 A) and C-C (1.513 A) bonds, unresolved because the real-space resolution
is about `pi/qmax` = 0.26 A, beating at their stretch periods. The slower
alternation near r = 2.2 to 2.8 A is the C...C distance (2.581 A) breathing at
the 847 cm^-1 skeletal mode. The faint structure above 3 A is the H...H
distances.

The inelastic panel is blank in every transform. In the independent atom model
the inelastic term is `sum_i S_i(q)`, which depends on the atom list and on q
but not on the geometry, so it is constant in time and the difference signal
removes it. It is not negligible in the raw intensity: 72 per cent of the total
at q = 8 A^-1. It simply does not modulate.

## Re-analysing without rerunning

The saved `.dat` can be re-transformed with a different window or over a
restricted q range:

```python
from analysis.scattering import reanalyze_scattering_short_time

reanalyze_scattering_short_time(
    "acetone_time_dependent_scattering_form_factors.dat",
    window_fs=100.0, hop_fs=25.0, q_range=(4.0, 12.0),
    max_frequency_cm1=3500.0, label="_highq",
)
```

## Files kept here

Only the `.png` figures are kept. The `.dat` grids are dense -- the pair
distribution alone is 1024 frames by 400 radial points, tens of megabytes in
total -- and they regenerate from the script in a few minutes, so they are
ignored rather than stored. Run the script to get them back.
