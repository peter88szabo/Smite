May, 2017

This README file contains:
1. Introduction to the package
2. Sample test
3. Input/Output
4. Major files and directories line-up
5. Contact


1. Introduction to the package:

This is package for the PES of CO2H2O two body described in the paper `Two-component, ab initio potential energy surface for CO2−H2O, extension to the hydrate clathrate, CO2@(H2O)20, and VSCF/VCI vibrational analyses of both`. 


The following is the simple abstract of the PES, see main article for more details.

The PES contains two sub-PESs, i.e., fit-SR and fit-LR, and they are connected by a switching function. 

Fit-SR: This is a fit based on 47573 2-b potentials calculated at CCSD(T)-F12b/aVTZ level.  It was fitted using a set of primary and secondary polynomials with highest order of 6 which result in 5835 coefficients. Fit rmse is 6.3 wavenumber.

Fit-LR: This is a fit based on 6244 2-b potentials calculated also at CCSD(T)-F12b/aVTZ level.  It was fit based on purified monomial symmetrization method with highest order of 4 which result in 109 coefficients. Fit rmse is 0.4 wavenumber. Fit-LR converges to 0 as monomer separates.

Overall PES: The switching occurs if the increased distance between Carbon and water Oxygen (R(CO)) from 5 Angstrom to 6 Angstrom.
 
 
2. Sample test: 

	Simply execute `./run_me_to_test.sh` can automatically compile all source codes and run the test program. 

	Or, alternatively, in order to compile and run manually:

        a. Compile all source code using provided `Makefile` with command 'make'. This will generate a executable file `pes.x`. The sample Makefile uses Intel Fortran (ifort) compiler.
        b. Execute the `pes.x` by command `./pes.x` to run the test program. By default it reads 'test_inpt.xyz' as input file. You can change this in 'main_test.f90'.



3. Input/Output:

In your own code, you should first initialize the PES once with `call pes_init_co2h2o_2b()` to read all coefficients, then you can use potential function `pot = f_co2h2o_2b(xx)`. In function, `pot` is the calculated potential in unit of Hartree, and `xx` is the input Cartesian coordinate, in unit of Bohr. Notice `xx` is a 2D array with dimension `xx(3,6)`, where 3 means x, y and z component respectively of the same atom and 6 is atoms in order `OOOHHC`.



4. Major files and directories line-up:

* pes_co2h2_sr.f90:
        This module contains fit-SR in the paper.

* pes_co2h2o_lr.f90:
        This module contains fit-LR in the paper.

* pes_co2h2o_lr_basis.f90:
        This module contains the polynomial used in the fit-LR.

* pes_co2h2o_2b.f90:
        This this is module of overall PES that combines fit-SR and fit-LR by a switching function.

* pes_main_test.f90:
        A sample main program to use PES.

* test_inp.xyz:
        This file contains the global minimum configuration of PES.

* coeff/:
        This folder contains all coefficients used in fit-LR and fit-SR.

* lib/:
        This folder contains library and mod required to compile fit-SR.

* run_me_to_test.sh:
		A simple executable script to walk you through PES test.


5. Contact:
Joel M. Bowman: jmbowma@emory.edu
Qingfeng (Kee) Wang: kee.wang@emory.edu
