import numpy as np

from utils.constants import R_GAS_HARTREE_PER_K


class NoseHoover:
    """
    Mass-aware single Nose-Hoover thermostat for Smite momenta.

    Smite stores Cartesian momenta p and computes kinetic energy as
    sum(p_i**2 / m_i) / 2.  The thermostat variable therefore couples to
    2*K - g*kT, not to p_i**2 - kT for unit masses.
    """

    def __init__(self, nfix, wmass, tau, target_temp):
        if tau <= 0.0:
            raise ValueError("Nose-Hoover thermostat coupling time must be positive")
        if target_temp <= 0.0:
            raise ValueError("Nose-Hoover thermostat temperature must be positive")

        self.nfix = int(nfix)
        self.wmass = np.asarray(wmass, dtype=float)
        self.ndof = len(self.wmass) - self.nfix
        if self.ndof <= 0:
            raise ValueError("Number of active degrees of freedom must be positive")

        self.kt = R_GAS_HARTREE_PER_K * target_temp
        self.qmass = self.ndof * self.kt * tau * tau
        self.xi = 0.0

    def _active_slice(self):
        if self.nfix == 0:
            return slice(None)
        return slice(0, -self.nfix)

    def kinetic_energy(self, p):
        active = self._active_slice()
        p_active = np.asarray(p, dtype=float)[active]
        wmass_active = self.wmass[active]
        return 0.5 * np.sum(p_active * p_active / wmass_active)

    def step(self, p, dt):
        p = np.asarray(p, dtype=float)
        active = self._active_slice()

        kinetic = self.kinetic_energy(p)
        self.xi += 0.5 * dt * (2.0 * kinetic - self.ndof * self.kt) / self.qmass

        p[active] *= np.exp(-self.xi * dt)

        kinetic = self.kinetic_energy(p)
        self.xi += 0.5 * dt * (2.0 * kinetic - self.ndof * self.kt) / self.qmass

        return p


def thermo_nosehoover(nfix, p, wmass, dt, tau, Ttarg, state=None):
    if state is None:
        state = NoseHoover(nfix=nfix, wmass=wmass, tau=tau, target_temp=Ttarg)
    return state.step(p, dt), state
