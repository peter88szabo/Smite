import numpy as np

class MDNose:
    def __init__(self):
        self.nc = 0
        self.pi = None
        self.xi = None
        self.nhq = 0.0
        self.onq = 0.0
        self.nhcham = 0.0

    def nhc_init(self, nchain, wopt, kt, ndim, seed=42):
        """
        Initialize the Nosé-Hoover chain.
        
        Parameters:
            nchain (int): Number of chains.
            wopt (float): Optimized frequency.
            kt (float): Boltzmann constant times temperature.
            ndim (int): Number of dimensions.
            seed (int): Seed for random number generator for reproducibility.
        """
        np.random.seed(seed)
        self.nc = nchain
        self.pi = np.zeros((self.nc, ndim))
        self.xi = np.zeros((self.nc, ndim))

        self.nhq = kt / (wopt ** 2)
        self.onq = 1.0 / self.nhq

        # Initialize `pi` with Gaussian-distributed random values
        self.pi = np.random.normal(0, np.sqrt(kt), (self.nc, ndim))
        self.nhcham = 0.0  # Reset the conserved quantity accumulator

    def nhc_step(self, p, fulldt, mts, kt, ndim):
        """
        Perform a Nosé-Hoover chain step.
        
        Parameters:
            p (numpy array): Momentum vector.
            fulldt (float): Full time step.
            mts (int): Number of multi-time step subdivisions.
            kt (float): Boltzmann constant times temperature.
            ndim (int): Number of dimensions.
        
        Returns:
            p (numpy array): Updated momentum vector.
        """
        dt = fulldt / mts
        dt2 = 0.5 * dt

        for j in range(ndim):
            for k in range(mts):
                # Evolve p
                b = self.onq * self.pi[0, j]
                a = b * dt2
                self.xi[0, j] += a
                p[j] *= np.exp(-a)

                # Half time-step on odd xi's and even pi's
                c = b * self.pi[0, j] - kt
                for i in range(2, self.nc, 2):
                    b = self.onq * self.pi[i, j]
                    a = b * dt2
                    self.xi[i, j] += a
                    if b == 0.0:
                        self.pi[i - 1, j] += c * dt2
                    else:
                        c = c / b
                        self.pi[i - 1, j] = c + (self.pi[i - 1, j] - c) * np.exp(-a)
                    c = b * self.pi[i, j] - kt

                if self.nc % 2 == 0:
                    self.pi[self.nc - 1, j] += c * dt2

                # Full time step on odd pi's and even xi's
                c = p[j] ** 2 - kt  # Assuming mass = 1
                if self.nc == 1:
                    self.pi[0, j] += c * dt
                else:
                    for i in range(1, self.nc, 2):
                        b = self.onq * self.pi[i, j]
                        a = b * dt
                        self.xi[i, j] += a
                        if b == 0.0:
                            self.pi[i - 1, j] += c * dt
                        else:
                            c = c / b
                            self.pi[i - 1, j] = (self.pi[i - 1, j] - c) * np.exp(-a) + c
                        c = b * self.pi[i, j] - kt

                    if self.nc % 2 != 0:
                        self.pi[self.nc - 1, j] += c * dt

                # Evolve p again
                b = self.onq * self.pi[0, j]
                a = b * dt2
                self.xi[0, j] += a
                p[j] *= np.exp(-a)

                # Half time-step on odd xi's and even pi's
                c = b * self.pi[0, j] - kt
                for i in range(2, self.nc, 2):
                    b = self.onq * self.pi[i, j]
                    a = b * dt2
                    self.xi[i, j] += a
                    if b == 0.0:
                        self.pi[i - 1, j] += c * dt2
                    else:
                        c = c / b
                        self.pi[i - 1, j] = c + (self.pi[i - 1, j] - c) * np.exp(-a)
                    c = b * self.pi[i, j] - kt

                if self.nc % 2 == 0:
                    self.pi[self.nc - 1, j] += c * dt2

        return p

    def nhc_cons(self, kt, ndim):
        """
        Compute the Nosé-Hoover chain conserved quantity.
        
        Parameters:
            kt (float): Boltzmann constant times temperature.
            ndim (int): Number of dimensions.
        
        Returns:
            nhcham (float): Conserved quantity for Nosé-Hoover chain.
        """
        self.nhcham = 0.0
        for j in range(ndim):
            for i in range(self.nc):
                self.nhcham += 0.5 * self.pi[i, j] ** 2 * self.onq + kt * self.xi[i, j]
        return self.nhcham

