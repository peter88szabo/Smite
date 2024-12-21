#!/usr/bin/env python 
# encoding: utf-8 
  
from PES import get_potential 
   
################################################################################ 
 
from PES import get_potential, initialize_potential
 
################################################################################
 
label = 'F + HD -> OH + D' 
 
reactants(  
    atoms = ['F', 'H', 'D'], 
    reactant1Atoms = [1], 
    reactant2Atoms = [2,3], 
    Rinf = (15,"bohr"), 
) 
 
transitionState( 
    geometry = ( 
        [[-3.10,   0.000,   0.000], 
         [0.000,    0.000,   0.000], 
         [1.43,    0.000,   0.000]], 
        "bohr", 
    ), 
    formingBonds = [(2,1)], 
    breakingBonds = [(3,2)], 
) 
 
#equivalentTransitionState( 
#    formingBonds=[(3,1)],  
#    breakingBonds=[(2,3)], 
#) 
 
thermostat('Andersen') 
 
################################################################################# 
 
xi_list = numpy.arange(-0.05, 1.05, 0.01) 
#xi_list = [-0.00] 
 
generateUmbrellaConfigurations( 
    dt = (0.0001,"ps"), 
    evolutionTime = (2,"ps"), 
    xi_list = xi_list, 
    kforce = 0.1 * T, 
) 
 
windows = [] 
for xi in numpy.arange(-0.05, 0.92, 0.01) : 
    window = Window(xi=xi, kforce=0.1*T, trajectories=100, equilibrationTime=(20,"ps"), evolutionTime=(100,"ps")) 
    windows.append(window) 

for xi in numpy.arange(0.96, 1.05, 0.01) : 
    window = Window(xi=xi, kforce=0.1*T, trajectories=100, equilibrationTime=(20,"ps"), evolutionTime=(100,"ps")) 
    windows.append(window) 

 
conductUmbrellaSampling( 
    dt = (0.0001,"ps"), 
    windows = windows, 
) 
 
computePotentialOfMeanForce(windows=windows, xi_min=-0.02, xi_max=1.06, bins=5000) 
 
 
computeRateCoefficient() 
 
