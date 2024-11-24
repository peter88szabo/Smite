import numpy as np
import random

def thermo_andersen(nfix, p, wmass, dt, prob, Ttarg):
    """
    Set atomic momenta to fulfill
    the Maxwell-Boltzmann distribution
    same procedure as to initialize the momenta from MB distribution

    The formula for velocities:
    v(i)=sqrt(RT/m(i))*N

    where N is a random number with normal distirbution

    But we calculate here momenta instead of velocities
    """

    RT = (8.3144598/1000.0/2625.5) * Ttarg  #Rgas in Hartree/K

    if random.uniform(0.0,1.0) < prob*dt :
       for i in range(len(p)-nfix):
           p[i] = np.sqrt(wmass[i] * RT) * random.normalvariate(mu=0.0, sigma=1.0) 

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


