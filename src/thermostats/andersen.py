import numpy as np
import random
from utils.constants import R_GAS_HARTREE_PER_K

def thermo_andersen(nfix, p, wmass, dt, collision_time, Ttarg):
    """
    Apply independent Poisson-distributed Andersen collisions to atoms.

    ``dt`` and the mean ``collision_time`` must use the same time unit.

    The formula for velocities:
    v(i)=sqrt(RT/m(i))*N

    where N is a random number with normal distirbution

    But we calculate here momenta instead of velocities
    """

    RT = R_GAS_HARTREE_PER_K * float(Ttarg)

    p = np.asarray(p, dtype=float)
    wmass = np.asarray(wmass, dtype=float)
    if p.shape != wmass.shape or p.ndim != 1 or len(p) % 3 != 0:
        raise ValueError("Andersen thermostat requires matching flat 3N p/mass arrays")
    if np.any(~np.isfinite(wmass)) or np.any(wmass <= 0.0):
        raise ValueError("Andersen thermostat masses must be finite and positive")
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("Andersen thermostat timestep must be finite and non-negative")
    if not np.isfinite(collision_time) or collision_time <= 0.0:
        raise ValueError("Andersen collision time must be finite and positive")
    if not np.isfinite(Ttarg) or Ttarg <= 0.0:
        raise ValueError("Andersen thermostat temperature must be finite and positive")
    removed_dof = int(nfix)
    if removed_dof < 0 or removed_dof >= len(p):
        raise ValueError("Andersen nfix must satisfy 0 <= nfix < 3N")

    collision_probability = -np.expm1(-float(dt) / float(collision_time))
    for atom_index in range(len(p) // 3):
        if random.uniform(0.0, 1.0) < collision_probability:
            component = slice(3 * atom_index, 3 * atom_index + 3)
            for index in range(component.start, component.stop):
                p[index] = np.sqrt(wmass[index] * RT) * random.normalvariate(
                    mu=0.0, sigma=1.0
                )

    return p



'''
 subroutine andersen(thermo, traj, freeze_me)
    ! Andersen thermostat
    type(nx_thermo_t), intent(in) :: thermo
    type(nx_traj_t), intent(inout) :: traj

    logical, dimension(:) :: freeze_me
    integer :: i, n

    real(dp) :: p, r, v_modulus, rg

    ! Probability P
    p = thermo%gamma * traj%dt

    do n = 1, size(traj%veloc, 2)
       if(freeze_me(n))then     !
          r = 1   ! Keep Newtonian velocity for this atom in each case!
          if (thermo%lvp >= 3) write(*, *) "  Atom ",n," is frozen!"
       else
          call random_number(r)  ! Random number uniform-distributed between 0 and 1
       end if
       !
       if     (p <  r) then
          if (thermo%lvp >= 3) write(*, *) "  Keep Newtonian velocity for atom:  ",n
       elseif (p >= r) then
          if (thermo%lvp >= 3) write(*, *) "  Random velocity generated for atom:",n
          v_modulus = dsqrt(BK * thermo%temp / traj%masses(n))
          do i = 1,3
             call gauss_rand(rg)
             traj%veloc(i, n) = v_modulus*rg
          enddo
       endif

    enddo

  end subroutine andersen
'''

