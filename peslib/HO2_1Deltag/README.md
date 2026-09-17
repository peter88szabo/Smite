# HO2 1-Delta-g analytical PES

This directory exposes the native excited-state HO2 potential as a Smite
`qchem="PES"` backend. It supports either H + O2 or OH + O atom ordering;
the Python interface reorders any H/O/O input internally and restores the
Cartesian gradient to the caller's atom order.

## Build

```bash
cd peslib/HO2_1Deltag
make
```

The build requires `gfortran` and produces `libho2_1deltag.so`. The runtime
files `ho2pes-excited.dat` and `para-ex.txt` must remain in this directory.

## Smite configuration

```python
qcinput = {
    "qchem": "PES",
    "pes_name": "HO2_1Deltag",
    "pes_path": "/path/to/Smite/peslib/HO2_1Deltag",
    "wfu": False,
}
```

Coordinates are passed by Smite in bohr. The C/Fortran wrapper evaluates the
native surface using its O-O, O-H, O-H distances in Angstrom and transforms
the native analytical derivatives into a nine-component Cartesian gradient
in Hartree/bohr. Smite obtains forces as the negative of that gradient.

The surface's energy zero is the native H + O2 asymptote. The original
source and tabulated data define its physical domain and extrapolation
behavior.
