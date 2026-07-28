module fh2_c_interface
   use, intrinsic :: iso_c_binding
   implicit none

   interface
      subroutine initialize_potential()
      end subroutine initialize_potential

      subroutine get_potential(q, natoms, nbeads, v, dvdq, info)
         import :: c_double, c_int
         integer(c_int), intent(in) :: natoms
         integer(c_int), intent(in) :: nbeads
         real(c_double), intent(in) :: q(3, natoms, nbeads)
         real(c_double), intent(out) :: v(nbeads)
         real(c_double), intent(out) :: dvdq(3, natoms, nbeads)
         integer(c_int), intent(out) :: info
      end subroutine get_potential
   end interface

contains

   subroutine fh2_init() bind(C, name="fh2_init")
      call initialize_potential()
   end subroutine fh2_init

   subroutine fh2_energy_gradient(q_bohr, natoms_in, energy, gradient, info) bind(C, name="fh2_energy_gradient")
      real(c_double), intent(in) :: q_bohr(*)
      integer(c_int), value, intent(in) :: natoms_in
      real(c_double), intent(out) :: energy
      real(c_double), intent(out) :: gradient(*)
      integer(c_int), intent(out) :: info

      integer(c_int) :: natoms, npoints
      real(c_double) :: q(3, 3, 1)
      real(c_double) :: v(1)
      real(c_double) :: dvdq(3, 3, 1)
      integer :: i, j

      info = 0
      energy = 0.0d0
      do i = 1, 9
         gradient(i) = 0.0d0
      end do

      if (natoms_in /= 3) then
         info = 1
         return
      end if

      do j = 1, 3
         do i = 1, 3
            q(i, j, 1) = q_bohr(3 * (j - 1) + i)
         end do
      end do

      natoms = 3
      npoints = 1
      v = 0.0d0
      dvdq = 0.0d0
      call get_potential(q, natoms, npoints, v, dvdq, info)

      if (info /= 0) return

      energy = v(1)
      do j = 1, 3
         do i = 1, 3
            gradient(3 * (j - 1) + i) = dvdq(i, j, 1)
         end do
      end do
   end subroutine fh2_energy_gradient

end module fh2_c_interface
