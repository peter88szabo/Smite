import numpy as np
from scipy.linalg import expm
from math import sqrt

class MDGLE:
    def __init__(self):
        # Initialize parameters and matrices as None or zero.
        self.gS = None
        self.gT = None
        self.gp = None
        self.ngp = None
        self.wnt = 0.0
        self.wns = 0.0
        self.langham = 0.0
        self.ns = 0

    def wn_init(self, dt, wopt, kt):
        """
        Initialize white-noise thermostat parameters.
        
        Parameters:
            dt (float): Time step.
            wopt (float): Optimized frequency.
            kt (float): Boltzmann constant times temperature.
        
        Initializes:
            wnt (float): White-noise thermostat parameter.
            wns (float): Standard deviation for noise.
            langham (float): Langevin conserved quantity accumulator.
        """
        g = 2.0 * wopt
        self.wnt = np.exp(-dt * g)
        self.wns = sqrt(kt * (1.0 - self.wnt ** 2))
        self.langham = 0.0  # Langevin conserved quantity accumulator

    def wn_step(self, p, ndim):
        """
        White-noise propagation step for momentum.
        
        Parameters:
            p (numpy array): Momentum vector.
            ndim (int): Number of dimensions.
        
        Returns:
            p (numpy array): Updated momentum vector.
        """
        for i in range(ndim):
            p[i] = self.wnt * p[i] + self.wns * np.random.normal()
        return p

    def gle_init(self, dt, wopt, kt, ndim):
        """
        Initialize GLE thermostat parameters.
        
        Parameters:
            dt (float): Time step.
            wopt (float): Optimized frequency.
            kt (float): Boltzmann constant times temperature.
            ndim (int): Number of dimensions.
        
        Initializes:
            gT (numpy array): Deterministic part of propagator.
            gS (numpy array): Stochastic part of propagator.
            gp (numpy array): Auxiliary momentum matrix.
        """
        print("Initialization of GLE thermostat.")
        
        # Load GLE-A matrix
        gA = np.loadtxt('GLE-A') * wopt
        self.ns = gA.shape[0] - 1

        # Load GLE-C matrix or initialize it as a diagonal matrix with kt
        try:
            gC = np.loadtxt('GLE-C')
            assert gC.shape[0] == gA.shape[0]
        except IOError:
            print("Using canonical-sampling, Cp=kT")
            gC = np.eye(self.ns + 1) * kt

        # Deterministic part of propagator
        self.gT = expm(-dt * gA)

        # Stochastic part
        gA_temp = gC - self.gT @ gC @ self.gT.T
        self.gS = self.cholesky_stabilized(gA_temp)

        # Initialize auxiliary vectors
        self.gp = np.zeros((ndim, self.ns + 1))
        for j in range(ndim):
            gr = np.random.normal(size=self.ns + 1)
            self.gp[j, :] = self.gS @ gr
        self.langham = 0.0

    def gle_step(self, p, ndim):
        """
        GLE propagation step.
        
        Parameters:
            p (numpy array): Momentum vector.
            ndim (int): Number of dimensions.
        
        Returns:
            p (numpy array): Updated momentum vector.
        """
        for j in range(ndim):
            self.gp[j, 0] = p[j]

        # Deterministic update
        self.ngp = self.gT @ self.gp.T
        self.ngp = self.ngp.T

        # Stochastic update
        for j in range(ndim):
            self.gp[j, :] = np.random.normal(size=self.ns + 1)
        self.ngp += self.gS @ self.gp.T
        self.gp = self.ngp.T

        # Update momentum
        for j in range(ndim):
            p[j] = self.gp[j, 0]
        
        return p

    def cholesky_stabilized(self, M):
        """
        Stabilized Cholesky decomposition with handling for negative eigenvalues.
        
        Parameters:
            M (numpy array): Matrix to decompose.
        
        Returns:
            L (numpy array): Lower-triangular Cholesky factor.
        """
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            L = np.zeros_like(M)
            for i in range(M.shape[0]):
                L[i, i] = sqrt(max(M[i, i], 0.0))
        return L

