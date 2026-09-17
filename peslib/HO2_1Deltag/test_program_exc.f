      program doit
      implicit real*8(a-h,o-z)
      dimension r(3),drv(3)

!     Constants and unit conversion factors:

!     [Anstrom]*c1=[bohr]
       c1=1.0d0/0.5291772d0
!     [kcal/mol]*c2=[Hartree]
       c2=1.d0/627.51d0
!     [g/mol]*c3=[electron mass unit]
       c3=1838.6836605d0
!     [Hartree]*c4=[eV]
       c4=27.2114
!     [Hartree]*c5=[cm-1]
       c5=219474.d0
!     [Hartree]*c7=[kJ/mol]
       c6=2625.5d0


      pi=4.d0*datan(1.d0)

      theta=110.d0
      theta=theta*pi/180.d0


!     distances in Angstrom
!     Energy in Hartree
      r0=0.6
      r(1)=1.24 

      do j=1,100 
          r(2)=r0+(j-1)*0.05

          r(3)=dsqrt(r(1)*r(1)+r(2)*r(2)-2.d0*r(1)*r(2)*dcos(theta))
          Vpes_exc=ho2excsmooth(0,r,drv)

          write(6,"(f10.3,1f12.5)") r(2),Vpes_exc*c4
      enddo

      stop "Scan is done"

      end program

