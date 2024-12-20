import ctypes
import numpy as np


'''
gfortran -shared -fPIC -o libarh2p_pes.so ArH2p_PES.f90
then you can use the python file
'''

# Load the shared library
ArH2 = ctypes.CDLL('./libarh2p_pes.so')

# Declare the init_pes subroutine signature (no return type for subroutines)
ArH2.init_pes.argtypes = []  # No arguments

# Declare the pes_gpr function signature
ArH2.pes_gpr.restype = ctypes.c_double                  # Return type is double
ArH2.pes_gpr.argtypes = [ctypes.c_double * 3]           # Input is an array of 3 doubles

# Call the function
ArH2.init_pes()
print("Initialization done")


# Example geometry input for pes_gpr
geom = np.array([1.0, 2.0, 3.0], dtype=np.float64)

# Call the pes_gpr function
geom_ctypes = (ctypes.c_double * 3)(*geom)  # Convert numpy array to ctypes array
pred = ArH2.pes_gpr(geom_ctypes)
print("Predicted PES:", pred)


