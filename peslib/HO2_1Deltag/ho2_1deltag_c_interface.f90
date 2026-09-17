module ho2_1deltag_c_interface
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic
   implicit none

   real(c_double), parameter :: bohr_to_angstrom = 0.52917706d0

   interface
      real(c_double) function ho2excsmooth(ider, ra, drv)
         import :: c_double, c_int
         integer(c_int) :: ider
         real(c_double) :: ra(3)
         real(c_double) :: drv(3)
      end function ho2excsmooth
   end interface

contains

   subroutine ho2_1deltag_init() bind(C, name="ho2_1deltag_init")
      real(c_double) :: ra(3), drv(3), energy

      ! Force the native spline tables to be read while Python has selected
      ! the PES directory as the process working directory.
      ra = (/1.30d0, 1.00d0, 1.60d0/)
      drv = 0.0d0
      energy = ho2excsmooth(0_c_int, ra, drv)
   end subroutine ho2_1deltag_init

   subroutine ho2_1deltag_energy_gradient(q_bohr, natoms, energy, gradient, info) &
         bind(C, name="ho2_1deltag_energy_gradient")
      real(c_double), intent(in) :: q_bohr(*)
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(out) :: energy
      real(c_double), intent(out) :: gradient(*)
      integer(c_int), intent(out) :: info

      real(c_double) :: ra(3), drv(3)
      integer :: i

      info = 0_c_int
      energy = 0.0d0
      do i = 1, 9
         gradient(i) = 0.0d0
      end do

      if (natoms /= 3_c_int) then
         info = 1_c_int
         return
      end if
      do i = 1, 9
         if (.not. ieee_is_finite(q_bohr(i))) then
            info = 3_c_int
            return
         end if
      end do

      ! The Python layer orders atoms as H, O(1), O(2).  The native surface
      ! uses the three permutation-invariant pair distances in Angstrom:
      ! ra = [R_OO, R_O(1)H, R_O(2)H].
      ra(1) = pair_distance_angstrom(q_bohr, 2, 3)
      ra(2) = pair_distance_angstrom(q_bohr, 2, 1)
      ra(3) = pair_distance_angstrom(q_bohr, 3, 1)
      if (minval(ra) <= 0.0d0) then
         info = 2_c_int
         return
      end if

      drv = 0.0d0
      energy = ho2excsmooth(1_c_int, ra, drv)
      if (.not. ieee_is_finite(energy) .or. any(.not. ieee_is_finite(drv))) then
         info = 3_c_int
         return
      end if

      ! ho2excsmooth returns dV/dR in Hartree/Angstrom.  The chain rule below
      ! produces dV/dq in Hartree/bohr for Smite's Cartesian coordinates.
      call add_pair_gradient(q_bohr, gradient, 2, 3, drv(1))
      call add_pair_gradient(q_bohr, gradient, 2, 1, drv(2))
      call add_pair_gradient(q_bohr, gradient, 3, 1, drv(3))
   end subroutine ho2_1deltag_energy_gradient

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

end module ho2_1deltag_c_interface
