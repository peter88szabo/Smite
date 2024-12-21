#!/bin/bash
for i in $(seq -0.05 0.01 1.05); do
echo '#!/usr/bin/env python ' > input"$i".py
echo '# encoding: utf-8 ' >> input"$i".py
echo '  ' >> input"$i".py
echo 'from PES import get_potential ' >> input"$i".py
echo '   ' >> input"$i".py
echo '################################################################################ ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'from PES import get_potential, initialize_potential' >> input"$i".py
echo ' '>> input"$i".py
echo '################################################################################' >> input"$i".py
echo ' '>> input"$i".py
echo "label = 'Cl + CH4 -> ClH + CH3' " >> input"$i".py
echo ' ' >> input"$i".py
echo 'reactants(  ' >> input"$i".py
echo "    atoms = ['H', 'C', 'H', 'H', 'H', 'Cl'], " >> input"$i".py
echo '    reactant1Atoms = [1,2,3,4,5], ' >> input"$i".py
echo '    reactant2Atoms = [6], ' >> input"$i".py
echo '    Rinf = (20 * 0.52918,"angstrom"), ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'transitionState( ' >> input"$i".py
echo '    geometry = ( ' >> input"$i".py
echo '        [[-4.62816774,    1.25360230,    0.95619134], ' >> input"$i".py
echo '         [-4.85261637,    2.15380812,    0.37457524], ' >> input"$i".py
echo '         [-4.27582601,    2.99669239,    0.76939671], ' >> input"$i".py
echo '         [-4.52244653,    1.94874800,   -0.91617734], ' >> input"$i".py
echo '         [-5.92205712,    2.37770299,    0.44663477], ' >> input"$i".py
echo '         [-4.18100977,    1.73669027,   -2.25097637]], ' >> input"$i".py
echo '        "angstrom", ' >> input"$i".py
echo '    ), ' >> input"$i".py
echo '    formingBonds = [(4,6)], ' >> input"$i".py
echo '    breakingBonds = [(2,4)], ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'equivalentTransitionState( ' >> input"$i".py
echo '    formingBonds=[(5,6)],  ' >> input"$i".py
echo '    breakingBonds=[(2,5)], ' >> input"$i".py
echo ') ' >> input"$i".py
echo 'equivalentTransitionState( ' >> input"$i".py
echo '    formingBonds=[(3,6)],  ' >> input"$i".py
echo '    breakingBonds=[(2,3)], ' >> input"$i".py
echo ') ' >> input"$i".py
echo 'equivalentTransitionState( ' >> input"$i".py
echo '    formingBonds=[(1,6)],  ' >> input"$i".py
echo '    breakingBonds=[(2,1)], ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
#echo "thermostat('GLE', A=('gle_A.txt','s^-1')) " >> input"$i".py
echo "thermostat('Andersen') " >> input"$i".py
echo ' ' >> input"$i".py
echo '################################################################################# ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'xi_list = numpy.arange(-0.05, 1.05, 0.01) ' >> input"$i".py
echo "#xi_list = [$i] " >> input"$i".py
echo ' ' >> input"$i".py
echo 'generateUmbrellaConfigurations( ' >> input"$i".py
echo '    dt = (0.0001,"ps"), ' >> input"$i".py
echo '    evolutionTime = (5,"ps"), ' >> input"$i".py
echo '    xi_list = xi_list, ' >> input"$i".py
echo '    kforce = 0.1 * T, ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
echo "xi_list = [$i] " >> input"$i".py
echo 'windows = [] ' >> input"$i".py
echo 'for xi in xi_list: ' >> input"$i".py
echo '    window = Window(xi=xi, kforce=0.1*T, trajectories=200, equilibrationTime=(200,"ps"), evolutionTime=(100,"ps")) ' >> input"$i".py
echo '    windows.append(window) ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'conductUmbrellaSampling( ' >> input"$i".py
echo '    dt = (0.0001,"ps"), ' >> input"$i".py
echo '    windows = windows, ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
echo '#computePotentialOfMeanForce(windows=windows, xi_min=-0.02, xi_max=1.10, bins=5000) ' >> input"$i".py
echo ' ' >> input"$i".py
echo '#computeRecrossingFactor( ' >> input"$i".py
echo '    #dt = (0.0001,"ps"), ' >> input"$i".py
echo '    #equilibrationTime = (20,"ps"), ' >> input"$i".py
echo '    #childTrajectories = 100000, ' >> input"$i".py
echo '    #childSamplingTime = (2,"ps"), ' >> input"$i".py
echo '    #childrenPerSampling = 100, ' >> input"$i".py
echo '    #childEvolutionTime = (0.05,"ps"), ' >> input"$i".py
echo '#) ' >> input"$i".py
echo ' ' >> input"$i".py
echo '#computeRateCoefficient() ' >> input"$i".py
echo ' ' >> input"$i".py

echo '#!/bin/bash ' > submit"$i".sh
echo "#$ -o stdout_1000_16_""$i"".txt" >> submit"$i".sh                                                                                                       
echo "#$ -e stderr_1000_16_""$i"".txt" >> submit"$i".sh                                                                                                       
echo '#$ -j y ' >> submit"$i".sh                                                                                                                      
echo "#$ -N ClCH41000K""$i"" " >> submit"$i".sh                                                                                                             
echo '#$ -l h_rt=9996:00:00 ' >> submit"$i".sh                                                                                                        
echo '#$ -l mem_total=1G' >> submit"$i".sh
echo '#$ -l long ' >> submit"$i".sh                                                                                                                  
echo '#$ -pe singlenode 4' >> submit"$i".sh 
echo 'cd ' >> submit"$i".sh
echo 'source .bash_profile ' >> submit"$i".sh
echo 'source /opt/intel/Compiler/11.0/074/bin/intel64/ifortvars_intel64.sh ' >> submit"$i".sh
echo 'cd /home/ysuleyma/RPMDrate/ ' >> submit"$i".sh
echo "python rpmdrate.py /home/ysuleyma/RPMDrate/Examples/Cl+CH4/input"$i".py 1000 16 -p 4" >> submit"$i".sh
echo ' ' >> submit"$i".sh
qsub submit"$i".sh 
done