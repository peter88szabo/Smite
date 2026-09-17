from enum import Enum

class QChem(Enum):
    XTB = 'XTB'
    ORCA = 'Orca'
    SPARROW_BIN = 'Sparrow_bin'
    SPARROW_PY = 'Sparrow_Py'
    PYSCF = 'PySCF'
    PSI4 = 'Psi4'
    MOLPRO = 'Molpro'
    PES = 'PES'

# General QCInput class for any quantum chemistry software
class Calculator:
    def __init__(self, qchem: QChem, **kwargs):
        self.qchem = qchem

        # Set default values using kwargs, or specific defaults based on qchem option
        self.charge = kwargs.get('charge', 0)
        self.multiplicity = kwargs.get('multiplicity', 1)
        self.wfu = kwargs.get('wfu', False)
        self.nproc = kwargs.get('nproc', 1)
        self.mem = kwargs.get('mem', 2000)
        self.path = kwargs.get('path', None)
        self.basis = kwargs.get('basis', "sto-3g")
        self.additional = kwargs.get('additional', None)
        self.details = kwargs.get('details', None)

        # Use match to set method based on the chosen qchem
        match self.qchem:
            case QChem.XTB:
                self.method = None
            case QChem.ORCA | QChem.MOLPRO | QChem.PYSCF | QChem.PSI4:
                self.method = "HF"
            case QChem.SPARROW_BIN | QChem.SPARROW_PY:
                self.method = "PM6"
            case QChem.PES:
                self.method = None
            case _:
                self.method = kwargs.get('method', None)  # Fallback to 'PBE' if no special case

    # Utility function to display the current setup
    def display(self):
        return (f"Calculator:\n qchem={self.qchem.value},\n method={self.method},\n charge={self.charge},\n multiplicity={self.multiplicity},\n "
                f"wfu={self.wfu},\n nproc={self.nproc},\n path={self.path},\n basis={self.basis},\n additional={self.additional}\n")


def display_calculators():
    # Helper for interactive inspection only; avoid side effects at import time.
    calc_xtb = Calculator(qchem=QChem.XTB)
    calc_orca = Calculator(qchem=QChem.ORCA)
    calc_sparrow = Calculator(qchem=QChem.SPARROW_BIN)
    calc_pyscf = Calculator(qchem=QChem.PYSCF)
    print(calc_xtb.display())
    print(calc_orca.display())
    print(calc_sparrow.display())
    print(calc_pyscf.display())
