module pes_co2h2o_2b
  use pes_co2h2o_lr
  use pes_co2h2o_sr
  implicit none

  real,parameter::auang=0.5291772083
  real,parameter::aucm=219474.6313710

contains
  subroutine pes_init_co2h2o_2b()
    call pes_init_co2h2o_sr()  
    call pes_init_co2h2o_lr()  
    write(*,*) 'PES-CO2-H2O-2b all initialized!'
  return
  end subroutine 

  real function f_co2h2o_2b(x)
  !=============================
  !Input: x(3,6) array
  !In order O O O H H C. First two O are for CO2.
  !Unit: Bohr
  !Output Unit: Hartree
  !=============================

    integer,parameter::wp=selected_real_kind(12,300)
    real(kind=wp),dimension(3,6)::x
    real(kind=wp)::rCO,s,p
    real(kind=wp),parameter::a = 5.0/auang
    real(kind=wp),parameter::b = 6.0/auang
    integer::units
    
    rCO = sqrt(abs((x(1,3)-x(1,6))**2+(x(2,3)-x(2,6))**2+(x(3,3)-x(3,6))**2)) 
      if (rCO <= a) then
        f_co2h2o_2b = f_co2h2o_sr(x)
      else if (rCO < b) then
        p = (rCO-a)/(b-a)
        s = 10*p**3 - 15*p**4 + 6*p**5
        f_co2h2o_2b= (1-s)*f_co2h2o_sr(x) + s*f_co2h2o_lr(x)
      else
      f_co2h2o_2b = f_co2h2o_lr(x)
      end if
   f_co2h2o_2b = f_co2h2o_2b
    return
  end function 
end module
