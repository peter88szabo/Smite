module ho2_c_interface
   use, intrinsic :: iso_c_binding
   implicit none

   real(c_double), parameter :: bohr_to_angstrom = 0.52917706d0

   interface
      real(c_double) function ho2gxpot(ider, ra, drv)
         import :: c_double, c_int
         integer(c_int) :: ider
         real(c_double) :: ra(3)
         real(c_double) :: drv(3)
      end function ho2gxpot
   end interface

contains

   subroutine ho2_init() bind(C, name="ho2_init")
      real(c_double) :: ra(3), drv(3), energy

      ra = (/1.2d0, 1.2d0, 1.0d0/)
      drv = 0.0d0
      energy = ho2gxpot(0, ra, drv)
   end subroutine ho2_init

   subroutine ho2_energy_gradient(q_bohr, natoms, energy, gradient, info) bind(C, name="ho2_energy_gradient")
      real(c_double), intent(in) :: q_bohr(*)
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(out) :: energy
      real(c_double), intent(out) :: gradient(*)
      integer(c_int), intent(out) :: info

      real(c_double) :: ra(3), drv(3)
      integer :: i

      info = 0
      energy = 0.0d0

      do i = 1, 9
         gradient(i) = 0.0d0
      end do

      if (natoms /= 3) then
         info = 1
         return
      end if

      ra(1) = pair_distance_angstrom(q_bohr, 2, 3)
      ra(2) = pair_distance_angstrom(q_bohr, 1, 2)
      ra(3) = pair_distance_angstrom(q_bohr, 1, 3)

      if (minval(ra) <= 0.0d0) then
         info = 2
         return
      end if

      drv = 0.0d0
      energy = ho2gxpot(1, ra, drv)

      call add_pair_gradient(q_bohr, gradient, 2, 3, drv(1))
      call add_pair_gradient(q_bohr, gradient, 1, 2, drv(2))
      call add_pair_gradient(q_bohr, gradient, 1, 3, drv(3))
   end subroutine ho2_energy_gradient

   real(c_double) function pair_distance_angstrom(q_bohr, iatom, jatom)
      real(c_double), intent(in) :: q_bohr(*)
      integer, intent(in) :: iatom, jatom
      real(c_double) :: dx, dy, dz

      dx = (q_bohr(3 * (iatom - 1) + 1) - q_bohr(3 * (jatom - 1) + 1)) * bohr_to_angstrom
      dy = (q_bohr(3 * (iatom - 1) + 2) - q_bohr(3 * (jatom - 1) + 2)) * bohr_to_angstrom
      dz = (q_bohr(3 * (iatom - 1) + 3) - q_bohr(3 * (jatom - 1) + 3)) * bohr_to_angstrom
      pair_distance_angstrom = sqrt(dx * dx + dy * dy + dz * dz)
   end function pair_distance_angstrom

   subroutine add_pair_gradient(q_bohr, gradient, iatom, jatom, dVdR_angstrom)
      real(c_double), intent(in) :: q_bohr(*)
      real(c_double), intent(inout) :: gradient(*)
      integer, intent(in) :: iatom, jatom
      real(c_double), intent(in) :: dVdR_angstrom

      real(c_double) :: dx, dy, dz, r_angstrom, scale

      dx = (q_bohr(3 * (iatom - 1) + 1) - q_bohr(3 * (jatom - 1) + 1)) * bohr_to_angstrom
      dy = (q_bohr(3 * (iatom - 1) + 2) - q_bohr(3 * (jatom - 1) + 2)) * bohr_to_angstrom
      dz = (q_bohr(3 * (iatom - 1) + 3) - q_bohr(3 * (jatom - 1) + 3)) * bohr_to_angstrom
      r_angstrom = sqrt(dx * dx + dy * dy + dz * dz)

      if (r_angstrom <= 0.0d0) return

      scale = dVdR_angstrom * bohr_to_angstrom / r_angstrom

      gradient(3 * (iatom - 1) + 1) = gradient(3 * (iatom - 1) + 1) + scale * dx
      gradient(3 * (iatom - 1) + 2) = gradient(3 * (iatom - 1) + 2) + scale * dy
      gradient(3 * (iatom - 1) + 3) = gradient(3 * (iatom - 1) + 3) + scale * dz

      gradient(3 * (jatom - 1) + 1) = gradient(3 * (jatom - 1) + 1) - scale * dx
      gradient(3 * (jatom - 1) + 2) = gradient(3 * (jatom - 1) + 2) - scale * dy
      gradient(3 * (jatom - 1) + 3) = gradient(3 * (jatom - 1) + 3) - scale * dz
   end subroutine add_pair_gradient

end module ho2_c_interface
