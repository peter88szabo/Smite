# MexicanHat.py 

#======================================================
# This script pretend to reproduce the result obtained 
# in the article 
#
#       Geometric phase effects in dynamics near conical intersections: 
#       Symmetry breaking and spatial localization
#       DOI:  10.1103/PhysRevLett.111.220406
# 
# and supplemental material therein
#======================================================

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
from random import seed 
import os 

np.random.seed(1234) # fix the seed 

def W_lower(x, y, a= 1., omega1= 1., omega2 = 1., c = .5, Delta = 0.):
# this function will be recalled from Epotential(q)
	V11 = omega1**2/2.*(x+a/2.)**2+omega2**2/2.*y*y+Delta/2.
	V12 = c*y
	V21 = V12
	V22 = omega1**2/2.*(x-a/2.)**2+omega2**2/2.*y*y-Delta/2.
	W_ = .5*(V11+V22)-.5*np.sqrt((V11-V22)**2+4*V12*V12)
	return W_

def W_upper(x, y, a= 1., omega1= 1., omega2 = 1., c = .5, Delta = 0.):
	V11 = omega1**2/2.*(x+a/2.)**2+omega2**2/2.*y*y+Delta/2.
	V12 = c*y
	V21 = V12
	V22 = omega1**2/2.*(x-a/2.)**2+omega2**2/2.*y*y-Delta/2.
	W = .5*(V11+V22)+.5*np.sqrt((V11-V22)**2+4*V12*V12)
	return W

def Epotential(q):
    return W_lower(q[0],q[1])
    
def Ep_sym(rho):
    return(.5*(rho-.5)**2)

n = 50
x = np.linspace(-2,2,n)
y = np.linspace(-2,2,n)

z = np.zeros((n,n))
z2 = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        # for j in range(n):
        z[i,j] = W_lower(x[i],y[j])
        z2[i,j] = W_upper(x[i],y[j])


#======================================================
# plotting the PES
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# X, Y = np.meshgrid(x,y)
# ax.plot_surface(X,Y,z)
# ax.plot_surface(X,Y,z2)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()
#======================================================

#======================================================
# With the gamma coefficient equal to 1 we have a 
# conical intersection in the point x=0 and y=0
#
# gamma = 2*c/(omega1*omega2*a)
#======================================================

#======================================================
# Now we want to introduce a semiclassical dynamics on 
# this surface running a classical particle with an 
# initial given energy
#
# Energy in the magnitude of 0.5-1.5 Joules are the 
# only ones interesting from this pov
#======================================================
n_dim = 2
dt = 0.05
ts = 5000
wmass = np.ones(2)
dq = 0.002
# =====================================================
# parameter for taking into account the topological 
# contribution of the geometric phase
l = 0
# =====================================================

# =====================================================
# The following block is only for the initialization of 
# all the random initial distribution of the states
#
# The reference paper for this Montecarlo type sampling
# is the following
# 
#       Unimolecular dissociation of methane: A trajectory study 
#       using Metropolis sampling 
# 
#       DOI: 10.1063/1.446715
# =====================================================

def init_initial_values():
    E = 1.
    q = np.array([0.5, 0.])
    E1 = E-Epotential(q)
    # 0.5*p**2/m = E1
    # for the moment, I will assume the initial movement is
    # only on the y direction
    p = np.array([0., np.sqrt(2.*E1)])
    return E, q, p

def stochastic_sampling(E, n_states, l=0., deltar=0.005, deltap=0.01, eps=0.15):
    # eps: epsilon is a parameter for chosing the proper energy shell
    #
    # l: l is the flag for accounting geometric phase effects or not
    Elim = 1.5*E
    rho = 1. 
    Ep = l/rho**2+Ep_sym(rho)
    if (E-Ep)<0:
        print('----Please choose a greater initial energy----')
    else:    
        Ek = E-Ep
    p = np.array([np.sqrt(Ek), np.sqrt(Ek)])
    Pold = 1. #parameter for kicking off the routine
    states = np.zeros((n_states,4))
    H= l/rho**2+Ep_sym(rho)+np.inner(p/2.,p)
    for i in range(n_states):
        xi = random.rand(1,n_dim+1)-.5
        rhon = rho + xi[0,0]*deltar
        pn = p + xi[0,1:]*deltap
        H_star = l/rho**2+Ep_sym(rho)+np.inner(p/2,p)
        if not (H_star > Elim):
            Pstar = eps/(eps**2+np.abs(H_star-H))
            if (Pstar/Pold>random.rand()):
                rho, p = rhon, pn
                Pold = Pstar
                H = H_star
        states[i,:] = [rho, p[0], p[1], H]
    return states

def state_generator(q, p, Pold, delta = 0.1, eps=0.001, Elim = 3):
	# routine for the implementation of the 
	# Metropolis algorithm for initial state configuration generator
    H = np.inner(p/2,p)+Epotential(q)
    xi = random.rand(1,2*n_dim)-.5
    qn = q + xi[0,:n_dim]*delta
    pn = p + xi[0,n_dim:]*delta
    H_star = np.dot(pn/2.,pn)+Epotential(qn)
    if not (H>Elim):
        Pstar = eps/(eps**2+np.abs(H_star-H))
        if (Pstar/Pold>random.rand()):
            q, p = qn, pn
            Pold = Pstar
    return q, p, Pold     

def initial_sampling(n_states=2000):
    states = np.zeros((n_states, 2*n_dim))
    E, q, p = init_initial_values()
    Pr = 1 #parameter for kicking off the routine
    for i in range(n_states):
        q, p, Pr = state_generator(q, p, Pr)
        states[i,:] = np.append(q,p) 
    return states        

#======================================================

def run_dynamics(E, q, p, l=0, m=1.):
    qt = np.zeros((n_dim, ts))
    pt = np.zeros((n_dim, ts))
    Et = np.zeros((3, ts))
    Ep = Epotential(q)
    Et[:,0] = np.array([np.dot(p/2.,p), Ep, Ep+np.dot(p/2., p)]) 
    for idim in range(n_dim):
        qt[idim,0], pt[idim,0] = q[idim], p[idim]
    #pt[:,0] = [p[0],p[1]]
    for i in range(ts-1):
        qt[:,i+1], pt[:,i+1] = vervelet(qt[:,i],pt[:,i])
        Ep = Epotential(qt[:, i+1])
        Ek = np.dot(pt[:,i+1]/2.,pt[:,i+1])
        Et[:,i+1] = np.array([Ek, Ep, Ep+Ek-l*1./(2.*LA.norm(qt[:,i+1]))]) 
    return qt, pt, Et

#======================================================
# The following function aims at implementing the idea 
# presented in the following artiche
#
#       An extension of the fewest switches surface hopping algorithm 
#       to complex Hamiltonians and photophysics in magnetic fields: 
#       Berry's phase and "magnetic" forces
# 
#       DOI:    10.1063/1.5088770 
#======================================================

def magnetic_force(q):
    # Routine for evaluating the fake magnetic force
    # provided by the Berry curvature 
    # 
    # 'flag' is a label for tuning the force
    # 1: the force is acting 
    # 0: the force is silent
    f = -0.5*q/(LA.norm(q))**3
    return f 

def force(q, l=1):
    # l: label for the topological contribution
    F = np.zeros(n_dim)
    tt = np.zeros(n_dim) # topological contribution
    # the topological Berry curvature gives a contribution equal to 1/R 2
    for i in range(0,n_dim):
        Dq = np.zeros(n_dim)
        Dq[i] = dq
        Vp1 = Epotential(q+Dq)
        Vm1 = Epotential(q-Dq)
        tt[i] = -.5*q[i]/(LA.norm(q))**3 #definition of the topological contribution
        F[i] = -.5*(Vp1-Vm1)/dq
    tt = magnetic_force(q)
    return F+l*tt

def time_evolution(q,p):
    q = q + p/wmass*dt
    p = p + .5*force(q)*dt
    return q, p

def vervelet(q,p):
    p = p + force(q,l)*dt*0.5
    q = q + p/wmass*dt
    p = p + force(q,l)*dt*0.5
    return q,p


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# dynamics
E, q, p = init_initial_values()


# fig = plt.figure()
# plt.plot(qt[0,:],qt[1,:])
# plt.plot(qt[0,0],qt[1,0],'r*')
# plt.plot(qt[0,999],qt[1,999],'b*')

# plt.show()

def show_initial_configurations():
    x = np.linspace(-2,2,1000)
    y = np.linspace(-2,2,1000)
    states = initial_sampling()
    q1 = states[:,0]
    q2 = states[:,1]
    fig = plt.figure()
    ax1 = fig.add_subplot(111,aspect='equal')
    ax1.plot(q1,q2,'*')
    ax1.plot(x,x)
    ax1.plot(x,np.zeros(1000))
    ax1.add_artist(plt.Circle((0,0),1,color='red'))
    plt.xlim([-0.5,1.5])
    plt.ylim([-1,1])
    plt.show()
    
def save_configurations_externally(states):
    if not os.path.exists("configurations/"):
        os.mkdir("configurations/")
    i = 0
    for row in states:
        if i%100 == 0:
            print(i)
        q, p = row[:n_dim], row[n_dim:]
        i = i + 1
        np.savetxt("configurations/%d.txt" %(i), np.append(q,p),fmt='%5.10f')
        # with open("configurations/%d.txt" %(i),"w+") as f:
        #    f.write("%d %d" %(q, p))
        #    f.close()     

def save_information(pt, qt, Et, i):
    if not os.path.exists("data/"):
        os.mkdir("data/")
    # with open("data/%d.out" %(i), 'w+') as f:
    #     f.write('%5.10f %5.10f %5.10f %5.10f %5.10f %5.10f %5.10f' %(q[0], q[1], p[0], p[1], E)  
    #     print('ok')
    # f.close()
    to_save = np.concatenate((qt, pt, Et), axis=0)
    np.savetxt('data/%d.out'%(i), to_save, fmt='%5.10f', newline='\r\n')

def evaluate_trajectories():
    if not os.path.exists("trajectories/"):
        os.mkdir("trajectories/")
    i = 0
    states = initial_sampling()        
    for row in states:
        q, p = row[:n_dim], row[n_dim:]
        E = np.dot(p/2.,p)+Epotential(q)
        qt, pt, Et = run_dynamics(E, q, p)
        i = i + 1
        # np.savetxt("trajectories/%d.txt" %(i), qt, fmt='%5.10f', newline='\n')
        save_information(pt, qt, Et, i)
        if i%250 == 0:
            print(i)
            plt.figure()
            plt.plot(qt[0,:],qt[1,:])
    plt.show()

def evaluate_trajectories_modified(states):
    i = 0
    for row in states:
        qp = np.fromstring(row, dtype= float, sep= ' ')
        q, p = qp[:n_dim], qp[n_dim:]
        E = np.dot(p/2.,p)+Epotential(q)
        # ----- Dynamics with no GP --------
        qt, pt, Et = run_dynamics(E, q, p, 0)
        # save information
        to_save = np.concatenate((qt, pt, Et), axis=0)
        np.savetxt('data/%d.out'%(i), to_save, fmt='%5.10f', newline='\r\n')
        # ----- Dynamics with  GP ----------
        qt, pt, Et = run_dynamics(E, q, p, 1)
        # save information
        to_save = np.concatenate((qt, pt, Et), axis=0)
        np.savetxt('data_GP/%d.out'%(i), to_save, fmt='%5.10f', newline='\r\n')
        i = i + 1

def create_initial_states_if_needed():
    if not os.path.exists('initial_values/'):
        os.mkdir('initial_values/')
        states = initial_sampling()
        np.savetxt('initial_values/states.txt', states, fmt='%5.10f', newline='\r\n')
    else:
        f = open('initial_values/states.txt', 'r')
        states = f.readlines() 
        states = [x.strip() for x in states]
    return states

def print_density(states):  
    Energies = states[:,-1]
    radii = states[:,0]
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(Energies, bins=100)
    plt.subplot(2,1,2)
    plt.hist(radii, bins=100)

if __name__ == "__main__":
    states = stochastic_sampling(2,80000) 
    print_density(states)
    states_GP = stochastic_sampling(2,80000,1)
    print_density(states_GP)
    plt.show()
    print("end reached")
