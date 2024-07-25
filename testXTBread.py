import re
import numpy as np

def parseXTB_energy(s):
    match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", s)

    if match:
        energy = float(match.group(1))
    else:
        print("Total energy not found in the XTB output.")

    return energy

def eatLines(f,num):
    for i in range(num):
        f.readline()

def parseXTB_grad(inputfile, natom):

    with open(inputfile, "r") as f:
         data = f.readlines()

    gradxyz = ''.join(data[(natom+2):(2*natom+2)])

       # Split the XYZ string into lines
    lines = gradxyz.strip().split('\n')
       # Count the number of lines that contain atomic coordinates
       #(excluding any empty or whitespace-only lines)

    grad = []
    # Loop through the lines and extract the gradient components
    for line in lines:
        parts = line.split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        grad.extend([x, y, z])

    return np.array(grad)


# Your output string
output = """
Topologies differ in total number of bonds
Writing topology from bond orders to xtbtopo.mol

MOs/occ written to file <molden.input>

molecular dipole:
                 x           y           z       tot (Debye)
 q only:        0.000       0.094      -0.002
   full:        0.066       0.023       0.123       0.358
molecular quadrupole (traceless):
                xx          xy          yy          xz          yz          zz
 q only:        0.388      -0.604      -0.089       0.406       0.087      -0.299
  q+dip:       -1.109      -1.840       1.770       1.154       0.420      -0.661
   full:       -0.747      -1.039       0.950       0.587       0.309      -0.203

           -------------------------------------------------
          | TOTAL ENERGY              -16.734816468299 Eh   |
          | GRADIENT NORM               0.199738631731 Eh/α |
          | HOMO-LUMO GAP               3.612574590570 eV   |
           -------------------------------------------------
"""


gradtext = """
$grad
  cycle =      1    SCF energy =   -16.81571636753   |dE/dxyz| =  0.013470
   -2.24005300223320      2.83527704724002     -0.33032223685753      C
    0.10790336171583      2.88304365449205      0.54893898303649      C
    2.23445185399983      1.26858637554057     -0.17060636425686      C
    2.23444996427370     -1.26858448581445     -0.17060825398298      C
    0.10790336171583     -2.88304176476593      0.54893709331037      C
   -2.24005867141157     -2.83527704724002     -0.33031656767915      C
    0.53922768048207      4.20681948150217      2.06043642381415      H
    4.05162950296996      2.20118133584976     -0.37654304813962      H
    4.05162761324384     -2.20118133584976     -0.37654871731799      H
    0.53923334966044     -4.20682137122829      2.06042886490965      H
   -3.72657015336950     -3.92397472379960      0.55631458410089      H
   -2.76298557434267     -1.74626189669151     -1.97033806164450      H
   -3.72657204309563      3.92397094434735      0.55630135601801      H
   -2.76297045653367      1.74626378641763     -1.97035128972737      H
   2.9627862199302E-05  -1.5402108325134E-04   3.3697939165758E-04
  -1.4134971895311E-03   2.3403319894592E-03  -3.8013569596380E-04
   3.2414913822499E-03   1.2165136569801E-03  -2.5225669960499E-03
   3.2412657728824E-03  -1.2163551059761E-03  -2.5223112685107E-03
  -1.4125064376575E-03  -2.3405551990259E-03  -3.7944979433186E-04
   2.8049014047183E-05   1.5445064189900E-04   3.3559342888476E-04
  -1.6405364878305E-03   1.5741902630379E-03   1.9625472021343E-04
   2.7969158427082E-03  -1.6450054416212E-03   1.3578963907070E-03
   2.7972439198295E-03   1.6448480604212E-03   1.3578616351726E-03
  -1.6405119963411E-03  -1.5739890759908E-03   1.9602096094125E-04
  -2.9905405887557E-03   1.2702195466006E-03   3.9131514781622E-03
  -2.2990251632301E-05   3.5703112733395E-03  -2.9008019581277E-03
  -2.9907418377760E-03  -1.2701556179950E-03   3.9130910173583E-03
  -2.3269004392412E-05  -3.5707839078772E-03  -2.9015833101132E-03
$end
"""
inputfile = 'gradient'
natom = 14

# Parse the output to extract the total energy
ene = parseXTB_energy(output)
grad = parseXTB_grad(inputfile, natom) 

print(ene,"\n")
print(gradtext)
print()
for i in range(natom):
    print(grad[3*i], grad[3*i+1], grad[3*i+2] )




