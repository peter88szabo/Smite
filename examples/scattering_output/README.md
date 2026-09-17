# Scattering example output

Reference output of the two scattering examples, kept so the plots can be
looked at without rerunning the trajectories.

| Directory | Example | System |
| --- | --- | --- |
| `ch2o/` | `example_scattering_form_factors.py` | formaldehyde, one dominant mode (C=O stretch, 1835 cm^-1) |

Acetone has its own directory, `examples/acetone_scattering/`, with the script
beside its figures.

Each run writes four pairs of files, all with elastic, inelastic and total side
by side:

| File | Contents |
| --- | --- |
| `*_time_dependent_scattering_form_factors` | `I(q, t)` |
| `*_scattering_power_spectrum` | Fourier transform along time |
| `*_scattering_pair_distribution` | sine transform along q, `dPDF(r, t)` |
| `*_scattering_short_time_spectrum` | windowed transform, `P(t, omega)` |

The `ch2o` directory also holds `_w60` and `_w120` short-time spectra, produced
by re-analysing the saved `.dat` with different windows rather than repeating
the trajectory.

Regenerate everything with:

```bash
python examples/example_scattering_acetone_multimode.py
```

Only the `.png` figures are kept. The `.dat` grids are dense and regenerate from
the examples in a few minutes, so they are ignored rather than stored.
