from enum import Enum

class QChem(Enum):
    XTB = 'XTB'
    ORCA = 'Orca'
    SPARROW_BIN = 'Sparrow_bin'
    SPARROW_PY = 'Sparrow_py'
    PYSCF = 'PySCF'
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
        self.path = kwargs.get('path', None)
        self.basis = kwargs.get('basis', "sto-3g")
        self.additional = kwargs.get('additional', None)

        # Use match to set method based on the chosen qchem
        match self.qchem:
            case QChem.XTB:
                self.method = None
            case QChem.ORCA | QChem.MOLPRO | QChem.PYSCF:
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


# Example usage:
calc_xtb = Calculator(qchem=QChem.XTB)
calc_orca = Calculator(qchem=QChem.ORCA)
calc_sparrow = Calculator(qchem=QChem.SPARROW_BIN)
calc_pyscf = Calculator(qchem=QChem.PYSCF)

# Function to print out the current configuration
def display_calculators():
    print(calc_xtb.display())
    print(calc_orca.display())
    print(calc_sparrow.display())
    print(calc_pyscf.display())

# Display the configurations
display_calculators()

