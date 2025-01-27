# Smite
Quasiclassical Trajectory Calculations (QCT) To Simulate Molecular/Surface Collisions
or Unimolecular Dynamics on Ab-initio or Analytical Potential Energy Surfaces


Install:
pip install .
or
pip install -e .


### Optional Dependencies ###
This package provides additional functionality with the following optional dependencies:

- pyscf: For quantum chemical calculations using the PySCF library.
- scine-sparrow: For quantum chemical calculations using SCINE Sparrow.

To install these options:
- Install with PySCF: pip install -e .[pyscf]
- Install with SCINE Sparrow: pip install -e .[sparrow]
- Install all options: pip install -e .[all]


or 
No external module: pip install Smite  
With PySCF: pip install Smite[pyscf]
With Scine-Sparrow: pip install Smite[sparrow]
With all External: pip install Smite[all]


to check the Pip list:
pip show Smite

to check the Python Path:
python -c "import smite; print(smite.__file__)"

