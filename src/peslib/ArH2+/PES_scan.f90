      program test
      implicit none
      integer :: i,j,k
      real*8, dimension(3) :: rr
      real*8 :: VHCN
      real*8 :: Vpes, Rjac, theta 
      real*8 :: r0, dr,dum1, dum2
      real*8 :: rCN, rHN, rHC, rho_C, rho_N
      real*8 :: xm0, xm1, gam
      real*8 :: pi, x, y, x0, y0, beta, alpha, gama

! r0: the HH' internuclear distance.
! r1: the ArH internuclear distance.
! r2: the ArH' internuclear distance.

      xm0=12.d0
      xm1=14.0030740048d0
      gam=xm1/(xm0+xm1)

      r0 = 1.2 
      dr = 0.1

      rCN=2.179

      do i=1,250
         Rjac = r0 + (i-1)*dr

         rho_C = rCN*gam
         rho_N = rCN-rho_C

         rHC = Rjac +rho_C 
         rHN = Rjac -rho_N 

         rr(1) = rCN  ! C-N
         rr(2) = rHC  ! H-C
         rr(3) = rHN  ! H-N

         call PES_GPR(rr,Vpes)

         write(69,"(f12.4,f16.9)") Rjac*0.5291772d0, Vpes
      enddo




      xm0=12.d0
      xm1=14.0030740048d0
      gam=xm1/(xm0+xm1)

      dr = 0.002

      x0 = -1.0
      y0 = 0.0
      rCN=2.179
      rho_C = rCN*gam
      rho_N = rCN-rho_C
 
      pi = 4.d0*atan(1.d0)

      do i=1,400
       ! x = x0 + (i-1)*dr
        y = y0 + (i-1)*dr
        do j=1,400
           x = x0 + (j-1)*dr
          !y = y0 + (j-1)*dr

          if(abs(x/y) .le. 1.0d0) then
            alpha  = acos(abs(x/y))
          elseif(abs(x/y) .gt. 1.d0) then
            alpha  = asin(abs(y/x))
          elseif(x .eq. 0.0d0 .and. y .eq. 0.d0) then
              stop "both zero"
          else
              alpha  = acos(abs(x/y))
          endif

          beta   = pi/2.0 - alpha
          gama   = pi/2.0 + alpha
          Rjac = sqrt(x*x + y*y)

          rHC = sqrt(Rjac**2 +rho_C**2 - 2.d0*Rjac*rho_C*cos(beta)) 
          rHN = sqrt(Rjac**2 +rho_N**2 - 2.d0*Rjac*rho_N*cos(gama)) 

          rr(1) = rCN  ! C-N
          rr(2) = rHC  ! H-C
          rr(3) = rHN  ! H-N

          call PES_GPR(rr,Vpes)

          write(70,"(2f10.3,1f16.9,5x,5f12.4)") y,x,Vpes,Rjac,rHC, rHN, &
          alpha*90.0/pi, x/y  
       enddo
      enddo

      end program


