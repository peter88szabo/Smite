program main
use pes_co2h2o_2b
implicit none

real,dimension(3,6)::xx,xxcw
real::pot
integer::i,ierr
character(len=2)::symbb(6)
character(len=32)::filename

call pes_init_co2h2o_2b()

filename = 'test_inp.xyz'
open(21,status='old',file=filename)
do
     read(21,*,iostat=ierr)
     if (ierr < 0) exit
     read(21,*)
     do i=1,6
        read(21,*) symbb(i),xx(:,i)
     end do
!Make sure the input length unit is Bohr and 
!order of input is OOOHHC in a (3,6) array
pot = f_co2h2o_2b(xx/auang)
write(*,*) 'Expected v2b in Hartree: -4.905147232641571E-003'
write(*,*) 'Calcualted v2b in Hartree ',pot
end do
close(11)
close(21)


end program
