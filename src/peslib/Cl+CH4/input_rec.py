#!/usr/bin/env python 
# encoding: utf-8 
  
from PES import get_potential 
   
################################################################################ 
 
from PES import get_potential, initialize_potential
 
################################################################################
 
label = 'Cl + CH4 -> ClH + CH3' 
 
reactants(  
    atoms = ['H', 'C', 'H', 'H', 'H', 'Cl'], 
    reactant1Atoms = [1,2,3,4,5], 
    reactant2Atoms = [6], 
    Rinf = (20 * 0.52918,"angstrom"), 
) 
 
transitionState( 
    geometry = ( 
        [[-4.62816774,    1.25360230,    0.95619134], 
         [-4.85261637,    2.15380812,    0.37457524], 
         [-4.27582601,    2.99669239,    0.76939671], 
         [-4.52244653,    1.94874800,   -0.91617734], 
         [-5.92205712,    2.37770299,    0.44663477], 
         [-4.18100977,    1.73669027,   -2.25097637]], 
        "angstrom", 
    ), 
    formingBonds = [(4,6)], 
    breakingBonds = [(2,4)], 
) 
 
equivalentTransitionState( 
    formingBonds=[(5,6)],  
    breakingBonds=[(2,5)], 
) 
equivalentTransitionState( 
    formingBonds=[(3,6)],  
    breakingBonds=[(2,3)], 
) 
equivalentTransitionState( 
    formingBonds=[(1,6)],  
    breakingBonds=[(2,1)], 
) 
 
thermostat('Andersen') 
 
################################################################################# 
 
xi_list = numpy.arange(-0.05, 1.05, 0.01) 
 
generateUmbrellaConfigurations( 
    dt = (0.0001,"ps"), 
    evolutionTime = (5,"ps"), 
    xi_list = xi_list, 
    kforce = 0.1 * T, 
) 
 
#xi_list = [-0.00] 
#windows = [] 
#for xi in xi_list: 
#    window = Window(xi=xi, kforce=0.1*T, trajectories=200, equilibrationTime=(200,"ps"), evolutionTime=(100,"ps")) 
#    windows.append(window) 
# 
#conductUmbrellaSampling( 
#    dt = (0.0001,"ps"), 
#    windows = windows, 
#) 
 
computePotentialOfMeanForce(xi_min=-0.02, xi_max=1.02, bins=5000) 
 
computeRecrossingFactor( 
    dt = (0.0001,"ps"), 
    equilibrationTime = (10,"fs"), 
    childTrajectories = 100000, 
    childSamplingTime = (2,"ps"), 
    childrenPerSampling = 100, 
    childEvolutionTime = (1.00,"ps"),
    xi_current = 0.9797,  
) 
 
computeRateCoefficient() 
 
