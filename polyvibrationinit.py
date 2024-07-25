import os
import numpy as np
import random
import math
from cenmass import cenmass
from euler import euler_rot
from thermal import thermal_vibr_mode  
from normalmode import getNormalmode, print_frequencies
from polyrotation import poly_rotation_init

#     [Anstrom]*c1=[bohr]
c1=1.0e0/0.5291772e0
#     [kcal/mol]*c2=[Hartree]
c2=1.e0/627.51e0
#     [g/mol]*c3=[electron mass unit]
c3=1838.6836605e0
#     [Hartree]*c4=[kcal/mol]
c4= 627.51
#     [Hartree]*c5=[cm-1]
c5=219474.e0
#     [femto-sec]*c6=[time in au]
c6=41.341105
#     [frequency in cm-1]*c9=[freq(bohr^(-1))]
c7 = 2625.5         # [Hartree] * c7 = [kJ/mol] 

c9=1.0e8*c1
#     [speed of light in atomic unit]
c10=137.035999074

Rgas = 8.3144598/1000.0/c7 #Hartree/K 

def polyatom_vibration_init(vibfile, RT, fix_vib, mass, q_eq, hessian):
    wmass = []   # auxiliary mass vector, same length as q and p
    for wx in mass:
        wmass += [wx]*3
    wmass = np.array(wmass)    


    ww, ww_low, L = getNormalmode(mass, hessian)

    ww_all = np.append(ww_low, ww)
    print_frequencies(ww_all)

    if os.path.exists(vibfile):
        os.remove(vibfile)
        print()
        print(f"{vibfile} already exisits, it has been deleted to create a new one.")
        print()

    with open(vibfile, 'w') as file:
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

        #file.write("%20s %15.4f \n" % ("Ezero [kcal/mol] = ", Ezero*c4, "Ezero [kJ/mol] = ", Ezero*c7) )
        #file.write("%20s %15.4f \n" % ("Evib  [kcal/mol] = ", Evib*c4), "Evib [kJ/mol]  = ", Evib*c7 )
        #file.write("%20s %15.4f \n" % ("Eexc  [kcal/mol] = ", (Evib-Ezero)*c4, "Eexc  [kJ/mol] = ", (Evib-Ezero)*c7) )

    q_norm = [a * math.cos(random.uniform(0, 2 * math.pi)) for a in ampl]
    v_norm = [-1.0 * ampl[i] * ww[i] * math.sin(random.uniform(0, 2 * math.pi)) for i in range(len(ampl))]

    L_tr = np.transpose(L)

    q_desc = np.transpose(q_eq) + np.matmul(L, np.transpose(np.array(q_norm)))
    v_desc = np.matmul(L, np.transpose(v_norm)) 
    p_desc = [wmass[i] * v_desc[i] for i in range(len(v_desc)) ] 

    q,p = cenmass(q_desc, p_desc, mass)


    #q,p, = poly_rotation_init(jrot, q_eq, mass, q, p)

    #Randomly roteate the molecule about its center of mass
    #q,p = euler_rot(q, p)
    return(q,p, ww_all)
