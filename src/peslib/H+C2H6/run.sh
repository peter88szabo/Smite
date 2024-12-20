for i in $(seq -0.05 0.01 1.01); do
echo '#!/usr/bin/env python ' > input"$i".py
echo '# encoding: utf-8 ' >> input"$i".py
echo '  ' >> input"$i".py
echo 'from PES import get_potential ' >> input"$i".py
echo '   ' >> input"$i".py
echo '################################################################################ ' >> input"$i".py
echo ' ' >> input"$i".py
echo "label = 'H + C2H6 -> H2 + C2H5' " >> input"$i".py
echo ' ' >> input"$i".py
echo 'reactants( ' >> input"$i".py
echo "    atoms = ['C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'H'], " >> input"$i".py
echo '    reactant1Atoms = [1,2,3,4,5,6,7,8], ' >> input"$i".py
echo '    reactant2Atoms = [9], ' >> input"$i".py
echo '    Rinf = (15 * 0.52918,"angstrom"), ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'transitionState( ' >> input"$i".py
echo '    geometry = (  ' >> input"$i".py
echo '        [[-0.18458289,    1.09944478,   -0.76346873], ' >> input"$i".py
echo '         [-0.09371221,    0.02584027,   -0.68919968], ' >> input"$i".py
echo '         [0.60997447,    1.48587575,   -1.38455625], ' >> input"$i".py
echo '         [-1.13897537,    1.35045534,   -1.20224965], ' >> input"$i".py
echo '         [-0.08879895,    1.70522604,    0.60541932], ' >> input"$i".py
echo '         [-0.16677838,    2.77846596,    0.51335803], ' >> input"$i".py
echo '         [-0.81879013,    1.34592969,    1.31564395], ' >> input"$i".py
echo '         [1.09626535,    1.34842526,    1.16925957], ' >> input"$i".py
echo '         [1.89966923,    1.10653532,    1.55151009]], ' >> input"$i".py
echo '        "angstrom"), ' >> input"$i".py
echo '    formingBonds = [(8,9)], ' >> input"$i".py
echo '    breakingBonds = [(5,8)], ' >> input"$i".py
echo ') ' >> input"$i".py
echo 'equivalentTransitionState( ' >> input"$i".py
echo '    formingBonds=[(6,9)],  ' >> input"$i".py
echo '    breakingBonds=[(5,6)], ' >> input"$i".py
echo ') ' >> input"$i".py
echo 'equivalentTransitionState( ' >> input"$i".py
echo '    formingBonds=[(7,9)],  ' >> input"$i".py
echo '    breakingBonds=[(5,7)], ' >> input"$i".py
echo ')  ' >> input"$i".py
echo ' ' >> input"$i".py
echo "thermostat('Andersen') " >> input"$i".py
echo ' ' >> input"$i".py
echo '################################################################################# ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'xi_list = numpy.arange(-0.05, 1.01, 0.01) ' >> input"$i".py
echo "#xi_list = [$i] " >> input"$i".py
echo ' ' >> input"$i".py
echo 'generateUmbrellaConfigurations( ' >> input"$i".py
echo '    dt = (0.0001,"ps"), ' >> input"$i".py
echo '    evolutionTime = (2,"ps"), ' >> input"$i".py
echo '    xi_list = xi_list, ' >> input"$i".py
echo '    kforce = 0.1 * T, ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
echo "xi_list = [$i] " >> input"$i".py
echo 'windows = [] ' >> input"$i".py
echo 'for xi in xi_list: ' >> input"$i".py
echo '    window = Window(xi=xi, kforce=0.1*T, trajectories=200, equilibrationTime=(20,"ps"), evolutionTime=(100,"ps")) ' >> input"$i".py
echo '    windows.append(window) ' >> input"$i".py
echo ' ' >> input"$i".py
echo 'conductUmbrellaSampling( ' >> input"$i".py
echo '    dt = (0.0001,"ps"), ' >> input"$i".py
echo '    windows = windows, ' >> input"$i".py
echo ') ' >> input"$i".py
echo ' ' >> input"$i".py
echo '#computePotentialOfMeanForce(windows=windows, xi_min=0.0, xi_max=1.1, bins=1000) ' >> input"$i".py
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
echo "#$ -o stdout_300_32_""$i"".txt" >> submit"$i".sh                                                                                                       
echo "#$ -e stderr_300_32_""$i"".txt" >> submit"$i".sh                                                                                                       
echo '#$ -j y ' >> submit"$i".sh                                                                                                                      
echo "#$ -N HC2H6""$i"" " >> submit"$i".sh                                                                                                             
echo '#$ -l h_rt=9996:00:00 ' >> submit"$i".sh                                                                                                        
echo '#$ -l mem_total=1G' >> submit"$i".sh
echo '#$ -l long ' >> submit"$i".sh                                                                                                                  
echo '#$ -pe singlenode 4' >> submit"$i".sh 
echo 'cd ' >> submit"$i".sh
echo 'source .bash_profile ' >> submit"$i".sh
echo 'source /opt/intel/Compiler/11.0/074/bin/intel64/ifortvars_intel64.sh ' >> submit"$i".sh
echo 'cd /home/ysuleyma/RPMDrate/ ' >> submit"$i".sh
echo "python rpmdrate.py /home/ysuleyma/RPMDrate/Examples/H+C2H6/input"$i".py 300 32 -p 4" >> submit"$i".sh
echo ' ' >> submit"$i".sh
qsub submit"$i".sh 
done