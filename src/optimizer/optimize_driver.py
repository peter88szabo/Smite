from orcarun     import Orca_Force, Orca_Energy
from pyscfrun    import PySCF_Force, PySCF_Energy
from sparrowbin  import Sparrowbin_Force, Sparrowbin_Energy
from sparrowpy   import SparrowPy_Force, SparrowPy_Energy
from xtbrun      import XTB_Force, XTB_Energy

import numpy as np

def Optimize(qcinput, file_wf, q, atoms):

    qchem = qcinput[0]

    if qchem == 'PySCF':
        print("No availble option yet")
        #V = PySCF_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Orca':
        V, geom = Orca_Optimize(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_Py':
        print("No availble option yet")
        #V = SparrowPy_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'Sparrow_bin':
        print("No availble option yet")
        #V = Sparrowbin_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'XTB':
        print("No availble option yet")
        #V = XTB_Energy(file_wf, q, atoms, qcinput)
    elif qchem == 'PES':
        print("No availble option yet")
    else:
        raise ValueError("Non-Existing Quantum Chemical method is input. Avaiable packages: Orca, PySCF, Sparrow_bin, Sparrow_Py, XTB")

    return V

#xyz ='''
#  O  -0.06783047125742      0.00000000000000     -0.04795183080185
#  H   0.03988406002555      0.00000000000000      0.96552726825997
#  H   0.92361641123187      0.00000000000000     -0.28423843745812
#'''
#c1 = 0.52917721092

#Natoms, atoms, q = parseXYZ(xyz)
#q = np.array(q) / c1


#charge = 0
#multiplicity = 1
#functional = 'b3lyp'
#base = 'pc-0'

#E_PySCF    = PySCF_Energy(q, atoms, charge, multiplicity, functional, base)
#grad_PySCF = PySCF_Force(q, atoms, charge, multiplicity, functional, base)


#method = 'DFTB3'
#method = 'PM6'
#E_Sparrow = Sparrow_Energy(q, atoms, charge, multiplicity, method)
#grad_Sparrow = Sparrow_Force(q, atoms, charge, multiplicity, method) 



#print("Energy = ", E_PySCF, E_Sparrow)
#print()

#for i in range(3*Natoms):
#    print("%4d %18.5e %18.5e" % (i, grad_PySCF[i], grad_Sparrow[i]))

#print()

