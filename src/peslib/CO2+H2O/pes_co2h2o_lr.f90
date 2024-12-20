module pes_co2h2o_lr
  use pes_co2h2o_lr_basis
  implicit none
  real::coeff(1:524)
  save coeff

contains
  !==================================!
  ! read the coefficients of the PES !
  !==================================!
  subroutine pes_init_co2h2o_lr()
  !======================
  !Input:xx(3,6)
  !In order: O O O H H C
  !Example, xx(:,4) is the coordiante of first H
  !Unit: Bohr
  !Output: Energy
  !Unit: Hartree
  !=====================

    integer::i
    
    open(10,file='./coeff/coeff_lr.dat',status='old')

    do i=1,size(coeff)
       read (10,*) coeff(i)
    end do
    write(*,*) 'PES-CO2-H2O-LR initializing...'
    return
    close (10)
  end subroutine

  !====================================!
  ! Function to evaluate the potential !
  !====================================!
  function f_co2h2o_lr(xyz)
    real,dimension(:,:),intent(in)::xyz
    real::f_co2h2o_lr
    !::::::::::::::::::::::::::::::
    real,dimension(size(xyz,2)*(size(xyz,2)-1)/2)::x
    real,dimension(3)::dr
    real,parameter::a0 = 7
    integer::i,j,k

    k = 1
    do i=1,size(xyz,2)-1
       do j=i+1,size(xyz,2)
          dr = xyz(:,i) - xyz(:,j)
          x(k) = sqrt(dot_product(dr,dr))
          k = k+1
       end do
    end do

    do i=1,size(x)
       x(i)=exp(-x(i)/a0)
    end do

    f_co2h2o_lr=emsav(x,coeff)

    return
  end function 
end module
