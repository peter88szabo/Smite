!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Symmetrized and smoothed isomerization barrier
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      double precision function ho2excsmooth(ider,rx,drv)
      implicit real*8(a-h,o-z) 
      dimension ra(3),rx(3),drv(3),drva(3),drvb(3),drvor(3)
      DATA angstrom/0.52917706d0/

       drOH=abs(rx(2)-rx(3))


       if(drOH .gt. 0.10) then
!         original symm PES:
          ho2excsmooth=ho2excpot(1,ider,rx,drvor)
          if (ider .eq. 1) then
            do i=1,3
              drv(i)=drvor(i)
            enddo
          endif

       else
!         original symm PES
          Vor=ho2excpot(1,ider,rx,drvor)
!         H-atom at O(a) on the PES
          ra(1)=rx(1)
          ra(2)=rx(2)
          ra(3)=rx(3)
          Va=ho2excpot(0,ider,ra,drva)

!         H-atom at O(b) on the PES
          ra(1)=rx(1)
          ra(2)=rx(3)
          ra(3)=rx(2)
          Vb=ho2excpot(0,ider,ra,drvb)

!         averaged PES:        
          Vav=(Va+Vb)/2.d0

          x=drOH*10.0
          SM=-x*x*x*x*(((20.0*x-70.0)*x+84.0)*x-35.0) 

          ho2excsmooth=Vav*(1.0-SM)+Vor*SM

          if (ider .eq. 1) then
            drv(1)=0.5d0*(drva(1)+drvb(1))
            drv(2)=0.5d0*(drva(2)+drvb(3))
            drv(3)=0.5d0*(drva(3)+drvb(2))

            do i=1,3
              drv(i)=drv(i)*(1.0d0-SM)+drvor(i)*SM
            enddo

            if (rx(2) .gt. rx(3)) then
              sgn=1.0d0
            elseif (rx(2) .lt. rx(3)) then
              sgn=-1.0d0
            else
              sgn=0.0d0
            endif
            dSMdr=140.0d0*x*x*x*(1.0d0-x)**3*10.0d0*sgn
            drv(2)=drv(2)+(Vor-Vav)*dSMdr
            drv(3)=drv(3)-(Vor-Vav)*dSMdr
          endif

          
       endif
      endfunction

c-----------------------------------------------------------------------
      double precision function ho2excpot(isym,ider,ra,drv)
c-----------------------------------------------------------------------
c     ra(1) = O-O, ra(2) = O-H, ra(3) = O-H  (angstrom)
c
cx   in this version the enrgy output is in hartrees, the derivatives
cx    in hartree/Angstroms
cx   the zero of energy is at the H+O2 limit by shifting the
cx   original energy by 0.0196725923 hartrees which is obtained 
cx   at ROO=1.2106 Angstrom and RO1H=RO2H=100.Angs
      implicit real*8(a-h,o-z)
cx    double precision, parameter :: pi=3.14159265358979311d0
cx    double precision, parameter :: angstrom=0.52917706d0
cx    double precision, parameter :: ev=27.211611d0
cx    double precision, parameter :: eref=2.85496548840d0
      DATA pi/3.14159265358979311d0/
      DATA angstrom/0.52917706d0/
      DATA ev/27.211611d0/
      DATA eref/2.85496548840d0/
      dimension ra(3),r(3),drv(3),dvdx(3),dtdr(3)
      logical first,linear 
      data first/.true./,delta/1.d-5/,eps/2.0d-14/
c
      if (first) then
	write(6,100)
	call potread
	first = .false.
      end if
c
      linear = .false.



!=======================================================
      r(1) = ra(1)

      if(isym .eq. 1) then
        r(2) = min(ra(2),ra(3))
        r(3) = max(ra(2),ra(3))
      else
!       for isomerization barrier:        
        r(2) = ra(2)
        r(3) = ra(3)
      endif
!=======================================================




c     if (r(2).ne.ra(2)) print *,'ho2excpot: echange R2 - R3'
      do 10 i=1,3
   10 r(i) = r(i)/angstrom
      costeta = (r(1)**2+r(2)**2-r(3)**2) / (2.d0*r(1)*r(2))
      gamma = abs(costeta) - 1.d0
c     write(6,105) costeta,gamma
      if (gamma.gt.eps) then
        costeta = sign(1.d0,costeta)
        write(6,103) costeta,r,r(1)+r(2)-r(3)
!!        stop
      elseif (abs(gamma).lt.eps) then
	costeta = sign(1.d0,costeta)
	linear = .true.
c       print *,'HO2GXPOT: linear =',linear
      end if
      call PESABC(r(1),r(2),costeta,epot,ider,dvdx)
cxpascho2excpot = epot*ev - eref
      ho2excpot = epot - eref/ev + 0.0196725923
      if (ider.ne.1) return
c
c     Jacobian Calculation.
c     If HO2 is linear, calculation of the second derivative (theta) 
c
      if (linear) then
	denomi = r(1)*r(2)*costeta
	teta = acos(costeta) + delta
	call ho2pes(r(1),r(2),cos(teta),epot,1,drv)
	dvdx(3) = costeta*drv(3)/delta
      else
        denomi = r(1)*r(2)*dsqrt(1.d0-costeta**2)
      end if
      dtdr(1) = (r(2)*costeta - r(1)) / denomi
      dtdr(2) = (r(1)*costeta - r(2)) / denomi
      dtdr(3) = r(3) / denomi
c     print *,'dtdr =',dtdr
c     print *,'HO2GXPOT: dvdx =',dvdx
c
c     transformation of derivatives in internuclear distances
c     the derivative (theta) is in radian**(-1)
c
      dvdx(3) = dvdx(3)*180.d0/pi
cx    drv(1) = (dvdx(1) + dtdr(1)*dvdx(3)) * ev/angstrom
cx    drv(2) = (dvdx(2) + dtdr(2)*dvdx(3)) * ev/angstrom
cx    drv(3) = dtdr(3)*dvdx(3) * ev/angstrom
      drv(1) = (dvdx(1) + dtdr(1)*dvdx(3)) /angstrom
      drv(2) = (dvdx(2) + dtdr(2)*dvdx(3)) /angstrom
      drv(3) = dtdr(3)*dvdx(3) /angstrom
      if (isym.eq.1.and.ra(3).lt.ra(2)) then
	tmp = drv(2)
	drv(2) = drv(3)
	drv(3) = tmp
      end if
c
      return
  100 format(/72('-')/5x,'Potentiel de Guo & Xie pour HO2',
     >        ' (avec les derivees analytiques)'/72('-')/)
  103 format(/'Erreur dans HO2GXPOT: cos(teta) =',f20.16,
     >      /'R =',3f20.16/', sum =',f20.16)
  105 format('costeta =',f20.16,', gamma =',e20.10)
      end
c     Potential energy surface for the excited electronic state of HO2
c     Please add the following statement in your main program:
c           call PREPESABC
c     use the following statement to calculate the PES when needed:
c           call PESABC(r1,r2,cth,vv,ider,drv)
c     where r1 and r2 are the two bondlength O-H in bohr.
c           cth is cosine value of the enclosed angle.
c           vv is the potential in hartree. 
c	ider=0, calculate potential only.
c	ider=1, calculate potential and derivatives together.
c       drv stands for the derivatives for three freedom of degree.
c	drv(1) is derivative for roo
c	drv(2) is derivative for roh
c	drv(3) is derivative for th

      subroutine PESABC(r1,r2,cth,vv,ider,drv)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      real*8 vv,drv(3)
      PI=3.14159265358979323846264338327950d0 
      ROO=r1
      ROH=r2
      ANG=dacos(CTH)*180.d0/PI
      IF(ROO .GT. 20d0) THEN
        IF (ROH.LT.3.20001.AND.ROH.GT.1.39999) THEN
          CALL READLR(ROO,ANG)
          CALL SPL1(ROH,Vv,1,drv)
        ELSE
        CALL HO2PES(ROO,ROH,CTH,Vv,ider,drv)

        ENDIF

      ELSE
        CALL HO2PES(ROO,ROH,CTH,Vv,ider,drv)
      ENDIF
     
cx    end subroutine PESABC
      return
      end


      
      SUBROUTINE READLR(R,ANG)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      PARAMETER(NA=9,NB=13)
      CHARACTER*80  STR
      DIMENSION PL(0:10),DPL(0:10),Y(NA),A(NA),Y1(NA),YTH(NA)
      COMMON /DATALR/ RL(NB),VL(NB),DV(NB),DTH(NB)
      DATA VMIN1/-150.7148428d0/           ! minimun of O+OH
!      DATA VMIN1/0.0d0/
      PI=DACOS(-1D0)
      do i=1,NA
      Y(i)=0.0D0
      Y1(i)=0.0D0
      YTH(i)=0.0D0
      end do

      OPEN(15,FILE='para-ex.txt')   !  READ PARAMETERS
      DO I=1,NB
      READ(15,'(A20,F6.3)')STR,RL(I)
      READ(15,'(A5,4F16.8)')STR,(A(JJ),JJ=1,4)
      READ(15,'(A5,4F16.8)')STR,(A(JJ),JJ=5,8)
      READ(15,'(A5,4F16.8)')STR,A(9)
      READ(15,*)
      READ(15,*)

        CTH=DCOS(ANG*PI/180D0)
        X2=CTH
      PL(0)=1.0d0
      PL(1)=X2
      DPL(0)=0.0d0
      DPL(1)=1.0d0
      DO  II=1,NA-1
        PL(II+1)=(X2*PL(II)*(2*II+1)-PL(II-1)*II)
     &             /(II+1.0d0)
        DPL(II+1)=((2*II+1)*(PL(II)+X2*DPL(II))
     &             -II*DPL(II-1))/(II+1.0d0)
      ENDDO
        DO IK=0,1
          GG=R**(IK+4)
          DO IJ=1,4
            FF=2*IJ-1
            F1=PL(IJ-1)/DSQRT(FF)
            Y(IJ+4*IK)=F1/GG
            Y1(IJ+4*IK)=(-(IK+4.0D0)*F1)/(GG*R)
            YTH(IJ+4*IK)=DPL(IJ-1)/DSQRT(FF)
     &             *(-DSIN(ANG*PI/180D0)*PI/180D0)/GG
          ENDDO
        ENDDO
          Y(9)=1
          Y1(9)=0
          YTH(9)=0
      SUM=0
      DSUM=0
      DTSUM=0
      DO IL=1,NA
        SUM=SUM+A(IL)*Y(IL)
        DSUM=DSUM+A(IL)*Y1(IL)
        DTSUM=DTSUM+A(IL)*YTH(IL)
      ENDDO

      VL(I)=SUM-VMIN1
      DV(I)=DSUM
      DTH(I)=DTSUM
      ENDDO  ! I
      CLOSE(15)
      return
      END

      SUBROUTINE SPL1(RA,V,NP,drv)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      PARAMETER(NB=13)
      COMMON /DATALR/ RL(NB),VL(NB),DV(NB),DTH(NB)
      PARAMETER (M=100,N=100,L=100)
      DIMENSION XT(M),Y(M),Y2(L),YR(M),YR2(L),YT(M),YT2(L)
      dimension drv(3)
      DATA DY1,DYN/1.0D30,1.0D30/
      NR1=0
      DO I=1,NB
        NR1=NR1+1
        XT(NR1)=RL(I)
        IF (NP.EQ.1)  Y(NR1)=VL(I)
        IF (NP.EQ.2)  Y(NR1)=DV(I)
      ENDDO
      CALL spline(xt,y,nr1,dy1,dyn,y2)
      do i=1,3
        drv(i)=0
      end do
      IF (NP.EQ.1) THEN
        CALL msplint(xt,y,y2,nr1,ra,V,A,B,C,D,H,KLO,KHI)
        DRV(2)=(Y(KHI)-Y(KLO))/H
     &       +((-3*A*A+1)*Y2(KLO)+(3*B*B-1)*Y2(KHI))*H/6.D0
        DO I=1,NB
          YR(I)=DV(I)
          YT(I)=DTH(I)
        ENDDO
        CALL spline(xt,yr,nr1,dy1,dyn,yr2)
        CALL splint(xt,yr,yr2,nr1,ra,drv(1))
        CALL spline(xt,yt,nr1,dy1,dyn,yt2)
        CALL splint(xt,yt,yt2,nr1,ra,drv(3))
      ELSE
        CALL splint(xt,y,y2,nr1,ra,V)
      ENDIF
      return
      END


      
      SUBROUTINE POTREAD                                                
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)                                
      PARAMETER (ipes=2)                                              
      PARAMETER (noo=28,noh=30,nth=30)                                
      PARAMETER (m=100,n=100,l=100)                                     
      dimension xt(m),y(m),y2(l)                                        
      data vmin1/-150.7148428d0/        ! minimun of O+OH       
!        data vmin1/0.0d0/                                              
!      data vasy/-150.48709294d0/                                      
      common /bvcut/vcut                                                
      common /pes/pesmin,roo(noo),roh(noh),thth(noo,noh,nth),         
     &  ve(noo,noh,nth),vs(noo,noh,nth),ind(noo,noh)                    
      PI=DACOS(-1D0)                              
        open(98,file='ho2pes-excited.dat')                            
        vcut=12.0d0/27.2114d0                                           
        pesmin=10000                                                    
        do i=1,noo                                                      
          do j=1,noh                                                    
            do k=1,nth                                                  
            read(98,*)ii,ra1,ra2,th1,ve1                      
            if (ra1+ra2+th1.eq.0)goto 11                                
cx          write(66,'(i7,3(2x,f7.3),3x,f15.8)')ii,ra1,ra2,th1,ve1   
            roo(i)=ra1                                                  
            roh(j)=ra2                                                  
            thth(i,j,k)=th1                                             
            ve(i,j,k)=ve1-vmin1                                         
            if (ve(i,j,k).gt.vcut) ve(i,j,k)=vcut                       
            if (ve(i,j,k).lt.pesmin) then                               
            pesmin=ve(i,j,k)                                            
            r1m=ra1                                                     
            r2m=ra2                                                     
            thm=th1                                                     
            endif                                                       
            enddo                                                       
11          continue                                                    
           kk=k-1                                                       
           kkk=kk                                                       
           ind(i,j)=kk                                                  
          enddo                                                         
        enddo                                                           
!        write(55,*)r1m,r2m,thm                                          
!        write(*,*)r1m,r2m,thm,pesmin                                    
!        write(*,*)' roo=',(roo(i),i=1,noo)                              
!        write(*,*)' roh=',(roh(i),i=1,noh)                              
!        write(*,*)' noo=',noo,' noh=',noh                               
       close(98)                                                        
 1001      format (1x,1f7.3,1f15.4,1f15.8,2f15.8)                       
        do i=1,noo                                                      
        do j=1,noh                                                      
         nh=0                                                           
        do k=1,ind(i,j)                                                 
      nh=nh+1                                                           
      xt(nh)=thth(i,j,k)                                                
      y(nh)=ve(i,j,k)                                                   
        enddo                                                           
      call spline(xt,y,nh,0d0,0d0,y2)                                   
        do ik=1,nh                                                      
        vs(i,j,ik)=y2(ik)                                               
!        write(101,1004)i,j,ik,vs(i,j,ik)                               
        enddo                                                           
        enddo                                                           
        enddo                                                           
1004    format(3i5,f15.8)                                               
      return
      END                                                              
                                                                        
                                                                        
                                                                        
      subroutine ho2pes(r10,r20,cthi,vpot,ider,drv)                                 
        implicit real*8(a-h,o-z)                                        
        parameter (rbohr=0.5291771)                                     
        parameter (pi=3.141592653589793d0)                              
       dimension drv(3)
      common /bvcut/vcut                                                
c : r1=r(O-O), r2=r(H-O)                                                
        r1=r10                                                          
        r2=r20                                                          
        cth=cthi                                                        
      r3=dsqrt(r1**2+r2**2-2*r1*r2*cth)        

      if (r3.lt.0.8d0.or.r2.lt.0.8d0) then                                 
      vpot=vcut                                                            
      return                                                            
      endif                                

      if (cth.gt.0.97d0.and.r2.lt.4d0.and.r1.lt.4d0)then
        if (r2.lt.r3) then
          cth=(r1**2+r3**2-r2**2)/(2*r1*r3)
          r2=r3
        endif
      else
!    modified by Peter 2015. 10. 19.
!        if (r2.gt.r3) then                                              
!        cth=(r1**2+r3**2-r2**2)/(2*r1*r3)                               
!        r2=r3                                                           
!        endif   
      endif
                                                                
                                                      
        th=dacos(cth)*180.0d0/pi                                        
        if (r1.lt.1.8d0) r1=1.8d0                                       
        if (r1.gt.20.0d0) r1=20.0d0
        if (r2.lt.0.8d0) r2=0.8d0                                     
        if (r2.gt.15.0d0) r2=15.0d0                                     
        CALL SPl3(r1,th,r2,vpot,ider,drv)                               
       if (r2.ge.15.0) then
	drv(2)=0.d0
      end if
      if (r1.ge.20.0) then
	drv(1)=0.d0
      end if
      if (vpot.gt.vcut) vpot=vcut
      return
      end                                                               
                                                                        
                                                                        
                                                                        
                                                                        
       subroutine spl3(r1,th,r2,vpot,ider,drv)                                  
       implicit real*8(a-h,o-z)                                         
        parameter (noo=28,noh=30,nth=30,m=50)                                
       dimension ya(m),yw(m),y2a(m),drv(3),dydr2(m),dy2dr2(m),dydth(m),
     $          dyadth(m),dy2dth(m),xt(m),y(m),y2(m)
        common /pes/pesmin,roo(noo),roh(noh),thth(noo,noh,nth),         
     &  ve(noo,noh,nth),vs(noo,noh,nth),ind(noo,noh)                    
       data dy1,dyn/1.0d30,1.0d30/                                      
       do 20 i=1,noo                                                    
       do 10 j=1,noh                                                    
       nthth=0                                                
       do 2 k=1,ind(i,j)
       nthth=nthth+1
       xt(nthth)=thth(i,j,k)
       y(nthth)=ve(i,j,k)
       y2(nthth)=vs(i,j,k)
   2   continue
        if (th.gt.xt(nthth)) th=xt(nthth)                             
        call msplint(xt,y,y2,nthth,th,ya(j),
     &a,b,c,d,h,klo,khi)
        if (ider.eq.1) then
            dd = (1.d0-3.d0*a*a)*y2(klo)+(3.d0*b*b-1.d0)*y2(khi)
            dyadth(j) = (y(khi)-y(klo))/h + dd*h/6.d0
          else
	    dyadth(j) = 0.d0
        end if
   10   continue                                                        
      nh=0     
      if (r1.gt.7.and.r2.gt.1.3999.and.r2.lt.3.00001) then
        do  j=2,14
         nh=nh+1
         xt(nh)=roh(j)
         y(nh)=ya(j)
        enddo
      else
        do  j=1,noh
         nh=nh+1
         xt(nh)=roh(j)
         y(nh)=ya(j)
        enddo
      endif

      call spline(xt,y,nh,dy1,dyn,y2a)   
       call msplint(xt,y,y2a,nh,r2,yw(i),a,b,c,d,h,klo,khi)
        if (ider.eq.1) then
	  call spline(xt,dyadth,nh,dy1,dyn,dy2dth)
          dd = (1.d0-3.d0*a*a)*y2a(klo) + (3.d0*b*b-1.d0)*y2a(khi) 
          dydr2(i) = (y(khi)-y(klo))/h + dd*h/6.d0
          dydth(i) = a*dyadth(klo) + b*dyadth(khi) + c*dy2dth(klo) 
     >               + d*dy2dth(khi)
        end if
   20 continue
      call splined(roo,yw,noo,dy1,dyn,y2a,r1,th,r2)                         
      call msplint(roo,yw,y2a,noo,r1,vpot,a,b,c,d,h,klo,khi)                                    
      if (ider.eq.1.and.r1.le.roo(noo)) then
        call spline(roo,dydr2,noo,dy1,dyn,dy2dr2)
        call spline(roo,dydth,noo,dy1,dyn,dy2dth)
        dd = (1.d0-3.d0*a*a)*y2a(klo) + (3.d0*b*b-1.d0)*y2a(khi) 
        drv(1) = (yw(khi)-yw(klo))/h + dd*h/6.d0
        drv(2) = a*dydr2(klo) + b*dydr2(khi) + c*dy2dr2(klo) 
     >           + d*dy2dr2(khi)
        drv(3) = a*dydth(klo) + b*dydth(khi) + c*dy2dth(klo) 
     >           + d*dy2dth(khi)
      end if
c
      return

       end                                                              
                                                                        
                                                                        
                                                                        
C##################################################################     
C# SPLINE ROUTINES                                                      
C#            Numerical recipes in fortran                              
C#            Cambrige University Press                                 
C#            York, 2nd edition, 1992.                                  
C##################################################################     
      SUBROUTINE splint(xa,ya,y2a,n,x,y)                                
      implicit double precision  (a-h,o-z)                              
      DIMENSION xa(n),y2a(n),ya(n)                                      
      klo=1                                                             
      khi=n                                                             
 1    if (khi-klo.gt.1) then                                            
        k=(khi+klo)/2                                                   
        if(xa(k).gt.x)then                                              
          khi=k                                                         
        else                                                            
          klo=k                                                         
        endif                                                           
      goto 1                                                            
      endif                                                             
      h=xa(khi)-xa(klo)                                                 
      if (h.eq.0.0d0) write(6,*) 'bad xa input in splint'               
      a=(xa(khi)-x)/h                                                   
      b=(x-xa(klo))/h                                                   
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**  
     *2)/6.0d0                                                          
      return                                                            
      END                                                               
C#######################################################################
      SUBROUTINE spline(x,y,n,yp1,ypn,y2)                               
      implicit double precision  (a-h,o-z)                              
      DIMENSION x(n),y(n),y2(n)                                         
      PARAMETER (NMAX=100)                                              
      DIMENSION u(NMAX)                                                 
      if (yp1.gt..99d30) then                                           
        y2(1)=0.0d0                                                     
        u(1)=0.0d0                                                      
      else                                                              
        y2(1)=-0.5d0                                                    
        u(1)=(3.0d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)          
      endif                                                             
      do 11 i=2,n-1                                                     
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))                               
        p=sig*y2(i-1)+2.0d0                                             
        y2(i)=(sig-1.0d0)/p                                             
        u(i)=(6.0d0*((y(i+1)-y(i))/(x(i+                                
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*        
     *u(i-1))/p                                                         
11    continue                                                          
      if (ypn.gt..99d30) then                                           
        qn=0.0d0                                                        
        un=0.0d0                                                        
      else                                                              
        qn=0.5d0                                                        
        un=(3.0d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))      
      endif                                                             
        y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.0d0)
      do 12 k=n-1,1,-1                                                  
        y2(k)=y2(k)*y2(k+1)+u(k)                                        
12    continue                                                          
      return                                                            
      END                                                               
                                                                        
C#######################################################################
      SUBROUTINE splined(x,y,n,yp1,ypn,y2,r1,th,r2)
      implicit double precision  (a-h,o-z)
      DIMENSION x(n),y(n),y2(n),drvlr(3)
      PARAMETER (NMAX=100)
      DIMENSION u(NMAX)
      if (yp1.gt..99d30) then
        y2(1)=0.0d0
        u(1)=0.0d0
      else
        y2(1)=-0.5d0
        u(1)=(3.0d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.0d0
        y2(i)=(sig-1.0d0)/p
        u(i)=(6.0d0*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
      if (ypn.gt..99d30) then
        qn=0.0d0
        un=0.0d0
      else
        qn=0.5d0
        un=(3.0d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
c------------------------------------------------------------
      if(r1.gt.19.99)then
        call readlr(r1,th)
        call spl1(r2,dv,2,drvlr)
        y2(n)=dv
      else
        y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.0d0)
!        write(*,*)y2(n)
      endif
c-----------------------------------------------------------
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      END
                                                                        
                                                                        

c-----------------------------------------------------------------------
      subroutine msplint(xa,ya,y2a,n,x,y,a,b,c,d,h,klo,khi)
c-----------------------------------------------------------------------
      implicit double precision  (a-h,o-z)
      dimension xa(n),y2a(n),ya(n)
c
      klo=1
      khi=n
 1    if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
        goto 1
      endif
c
      h = xa(khi)-xa(klo)
      if (h.eq.0.0d0) write(6,*) 'bad xa input in splint'
      a = (xa(khi)-x)/h
      b = (x-xa(klo))/h
      c = (a**3-a)*h*h/6.d0
      d = (b**3-b)*h*h/6.d0
      y = a*ya(klo) + b*ya(khi) + c*y2a(klo) + d*y2a(khi)
c
      return
      end
c
