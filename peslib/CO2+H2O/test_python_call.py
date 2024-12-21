import ctypes
import numpy as np

# Constants
auang = 0.5291772083  # Conversion factor for Bohr to Angstrom
filename = 'test_inp.xyz'  # Input file

# Load the shared library
libpes = ctypes.CDLL('./libpesco2h2o.so')

# Initialize Fortran subroutine and function
#libpes.pes_init_co2h2o_2b.argtypes = []
#libpes.pes_init_co2h2o_2b.restype = None

#libpes.f_co2h2o_2b.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, shape=(3, 6))]
#libpes.f_co2h2o_2b.restype = ctypes.c_double

# Initialize PES
#libpes.pes_init_co2h2o_2b()


'''
# Read the input file
xx = np.zeros((3, 6), dtype=np.float64)  # Array for coordinates
symbb = [''] * 6  # Atomic symbols

with open(filename, 'r') as f:
    while True:
        try:
            f.readline()  # Skip line 1 (number of atoms)
            f.readline()  # Skip line 2 (comment)
            for i in range(6):
                line = f.readline().split()
                symbb[i] = line[0]
                xx[:, i] = list(map(float, line[1:4]))  # Read x, y, z
            # Ensure coordinates are in Bohr
            xx_bohr = xx / auang

            # Call Fortran function
            pot = libpes.f_co2h2o_2b(xx_bohr)

            # Output results
            print(f"Expected v2b in Hartree: -4.905147232641571E-003")
            print(f"Calculated v2b in Hartree: {pot}")
        except ValueError:
            break  # End of file

print("Calculation completed.")


filename = 'test_inp.xyz'
open(21,status='old',file=filename)
do
     read(21,*,iostat=ierr)
     if (ierr < 0) exit
     read(21,*)
     do i=1,6
        read(21,*) symbb(i),xx(:,i)
     end do
!Make sure the input length unit is Bohr and 
!order of input is OOOHHC in a (3,6) array
pot = f_co2h2o_2b(xx/auang)
write(*,*) 'Expected v2b in Hartree: -4.905147232641571E-003'
write(*,*) 'Calcualted v2b in Hartree ',pot
end do
close(11)
close(21)


end program
'''
