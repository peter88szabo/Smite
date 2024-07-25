import numpy as np
import random
import math
from cenmass import cenmassQ
from normalmode import getNormalmode, print_frequencies

#     [Anstrom]*c1=[bohr]
c1=1.0e0/0.5291772e0
#     [Hartree]*c4=[kcal/mol]
c4= 627.51
#     [Hartree]*c5=[cm-1]
c5=219474.e0
#     [frequency in cm-1]*c9=[freq(bohr^(-1))]
c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 

c9=1.0e8*c1
#     [speed of light in atomic unit]
c10=137.035999074

Rgas = 8.3144598/1000.0/c7 #Hartree/K 


#-------------------------------------------------------
def thermal_vibr_mode(RT,ome):
#-------------------------------------------------------

    rand = random.uniform(0,1)

    vib = -RT*np.log(1.0-rand)/abs(ome)

    nvib = int(vib) #harmonic oscillator quantum number 

    return nvib
#-------------------------------------------------------


#-------------------------------------------------------------------------------------------
def polyatom_vibration_init(vibfile, phase_sampling, temp, fix_vib, mass, q_eq, hessian):
#-------------------------------------------------------------------------------------------

    RT = Rgas * temp

    ww, ww_low, L = getNormalmode(mass, hessian)

    ww_all = np.append(ww_low, ww)
    #print_frequencies(ww_all)

    ##### rotors[index of internal rotor][atomA][atomB] defines the axis and here
    ##### we can assign the clusters on the two side of the rotor-axis
    #####
    ##### Here we should project out the possible internal rotors, and sample them
    ##### as a free rotor
    ##### here we calcualte the vibrational freqs as well as the eigenvectors


    ##### First we do the pure vibrational sampling witht the truncated normal modes
    ##### then we applied the rotations along all axis




    ##### Create a dictionary with 1st columns as the indices of modes
    ##### then some modes could be signed as with Q, T, E, R to give energy, tempreture or quantum, rotor to that mode
    ###### ['1', 'Q', '0'] 
    ###### ['2', 'Q', '3']
    ###### ['3', 'T', '500.0']
    ###### ['4', 'T', '300.0']
    ###### ['5', 'E', '30.0']
    ###### ['6', 'E', '11.0']
    ###### ['7', 'R', '360.0']
    ###### ['8', 'R', '180.0']


    # we need also a function that can automatically create all modes methods to be the same for all
    # and we need also another one that checks that Q, E, R, T values are given nothing else

    #or just we use Gunnar Nyman's method to sample all modes with respect to a total Energy and J


    # for all modes that are not rotors (or even rotors but with 0 energy or ZPE) we do normal mode sampling
    # then after this we sample one-by-one the interl rotors too
    # then discard the geoms where there is some overlap between atoms




###### built in diatomic constants
###### small library for basic polyatomic molecules (freq): 
# H2O, HO2, H3O+, O3, CO2, H2CO3, CH2, H2S, HSO, HSO, HNO, HCN, HNC, CH3, CH4, NO2, NO3, NH2, NH3, NH4+, H2CO, H2COO []criegee,, HCOOH, HOCO, CH3-OH, CH3-NH2, CH3-NO2, CH3-SO2, CH3-SH  
#benzene, methyil benzene, CN-bene, OH-benzene, NH2-benzene, isoprene, 



#give the possibility to define a list of stationary point for unimolecular samplin, and do all of them all at once
#one-by-one HEssian calculation then sampling for each
#Only an extra function is needed witha loop over sturctures
#
#here is the only importance is telling the code that its an Equlbirum or TS structure. For TS e should sample the imaginary mode too...
## TEST the imaginary mode sampling




    with open(vibfile, 'a') as file:
        file.write("++++++Thermal sampled vib modes++++++ \n")
        file.write("%20s %10.1f \n" % ("Temperature [K] = ", RT/Rgas))
        file.write("%6s %6s %12s %8s\n" % ("index1","index2", "freq[cm-1]", "    nvib"))
        nvib = []
        for i in range(len(ww)): 
            if i == fix_vib[0]:
                nv = fix_vib[1] #value of fixed quantum number
                nvib += [nv]
                file.write("%6d %6d %12.1f %8d %13s\n" % (i+6, i, ww[i]/c10*c9/(math.pi * 2), nv, " <-- fix mode"))
            else:
                nv = thermal_vibr_mode(RT,ww[i])
                nvib += [nv]
                file.write("%6d %6d %12.1f %8d\n" % (i+6, i, ww[i]/c10*c9/(math.pi * 2), nv))
        file.write("---------------------------------------\n")


        energy = [ww[i]*(nvib[i] + 0.5) for i in range(len(ww))]
        ampl = [math.sqrt(2.0 * energy[i])/ww[i] for i in range(len(ww))]

        Evib = sum(np.array(energy))
        Ezero = 0.5*sum(np.array(ww))
   
        file.write("{:<21} {:11.3f} {:<25} {:11.3f}\n".format("Ezero [kcal/mol] =", Ezero*c4, "   Ezero [kJ/mol] =", Ezero*c7))
        file.write("{:<21} {:11.3f} {:<25} {:11.3f}\n".format("Evib  [kcal/mol] =", Evib*c4,  "   Evib  [kJ/mol] =", Evib*c7))
        file.write("{:<21} {:11.3f} {:<25} {:11.3f}\n".format("Eexc  [kcal/mol] =", (Evib-Ezero)*c4,  "   Eexc  [kJ/mol] =", (Evib-Ezero)*c7))


    if phase_sampling == 'cosine':
        q_norm = [a * math.cos(random.uniform(0, 2 * math.pi)) for a in ampl]
    elif phase_sampling == 'linear':
        q_norm = [a * random.uniform(-1, 1) for a in ampl]
    else:
        raise ValueError("Your given {sampling} method is not available. Options to choose: cosine or linear\n")

    q_desc = np.transpose(q_eq) + np.matmul(L, np.transpose(np.array(q_norm)))

    q = cenmassQ(q_desc,  mass)


    return(q)
#-------------------------------------------------------------------------------------------
