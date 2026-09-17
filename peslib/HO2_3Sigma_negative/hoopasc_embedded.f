C
C         CALCULATE THE HO2 RKHS POTENTIAL ENERGY AND DERIVATIVE
C  modeled after the embedding of the DMBE IV PES into VENUS96
C
      SUBROUTINE DMBE
      implicit real*8(a-h,o-z)
      PARAMETER(ND1=100,NDP=10)
      COMMON/QPDOT/Q(150),PDOT(150)
      COMMON/PQDOT/P(150),QDOT(150),W(50)
cxdmbeCOMMON/QPDOT/Q(3*ND1),PDOT(3*ND1)
cxdmbeCOMMON/PQDOT/P(3*ND1),QDOT(3*ND1),W(ND1)
      COMMON/FORCES/NATOMS,I3N,NST,NM,NB,NA,NLJ,NTAU,NEXP,NGHOST,
     *NTET,NVRR,NVRT,NVTT,NANG,NAXT,NSN2,NRYD,NHFD,NLEPSA,NLEPSB,NDMBE
      COMMON/CONSTN/C1,C2,C3,C4,C5,C6,C7,PI,HALFPI,TWOPI
      COMMON/DMBEB/VHOO,NDMB(3)
cxdmbeDIMENSION RR(3),GRAD(3)
      COMMON /HO2_bondlength/ RR(3),GRAD(3)
      LOGICAL FIRST
      DATA FIRST/.TRUE./CA0,CA1/0.5291772D0,627.51D0/
      SAVE FIRST
C
C         SCREEN FOR ATOM LABELS
C
      IF (FIRST) THEN
         DO 20 I=1,NATOMS
            TMP=DABS(1.0D0-W(I)/1.0D0)
            IF (TMP.LT.0.05D0) NDMB(3)=I
cx    write(6,*) '+++ TMP',TMP,NDMB(3),(W(m),m=1,3)
   20    CONTINUE
         K=1
         DO 40 I=1,NATOMS
            TMP=DABS(1.0D0-W(I)/16.0D0)
            IF (TMP.LT.0.05D0.AND.K.LE.2) THEN
               NDMB(K)=I
               K=K+1
cx    write(6,*) '+++ TMP1',TMP,I,NDMB,K
            ENDIF
   40    CONTINUE
         IDUM=NDMB(1)*NDMB(2)*NDMB(3)
         IF (IDUM.EQ.0) STOP
         FIRST=.FALSE.
      ENDIF
C
C         INTERATOMIC DISTANCES IN ATOMIC UNITS
C
      DO 60 J=1,3
         DO 50 K=J+1,3
            JK=(J-1)*(2*3-J)/2+K-J
            J3=3*NDMB(J)
            J2=J3-1
            J1=J2-1
            K3=3*NDMB(K)
            K2=K3-1
            K1=K2-1
            T1=Q(J1)-Q(K1)
            T2=Q(J2)-Q(K2)
            T3=Q(J3)-Q(K3)
cxpasc      RR(JK)=DSQRT(T1*T1+T2*T2+T3*T3)/CA0
            RR(JK)=DSQRT(T1*T1+T2*T2+T3*T3)
   50    CONTINUE
   60 CONTINUE
C
C         ENERGY AND DERIVATIVES WITH RESPECT TO
C         INTERATOMIC DISTANCES
C
cxdme DMBE assignment: R1=X(1) R2=X(2) R3=X(3) F=VOO(R1)+VOH(R2)+VOH(R3)+THREBQ(Q1,Q2,Q3)+
cho2_pasc assignment     ra(1) = O-O, ra(2) = O-H, ra(3) = O-H  (angstrom)
C
cxdmbeCALL HO2SUR(RR,VDMBE)
cxdmbeCALL HO2DER(RR,GRAD)
      VHOO=ho2gxpot(1,RR,GRAD)
cx    write(35,'(7f15.8)') RR,VHOO,GRAD
      CC1=CA1*C1
      CC0=CC1/CA0/CA0
cxpasc from ho2gxpot energy comes in hartrees, derivatives in hartrees/Angstroms
      VHOO=VHOO*CC1
      DO 70 I=1,3
cxpasc   GRAD(I)=GRAD(I)/RR(I)*CC0
         GRAD(I)=GRAD(I)/RR(I)*CC1
   70 CONTINUE
C
C        CALCULATE (DV/DQ)'S
C
      DO 90 J=1,3
         DO 80 K=J+1,3
            JK=(J-1)*(2*3-J)/2+K-J
            J3=3*NDMB(J)
            J2=J3-1
            J1=J2-1
            K3=3*NDMB(K)
            K2=K3-1
            K1=K2-1
            T1=Q(J1)-Q(K1)
            T2=Q(J2)-Q(K2)
            T3=Q(J3)-Q(K3)
            TMP=GRAD(JK)*T1
            PDOT(K1)=PDOT(K1)-TMP
            PDOT(J1)=PDOT(J1)+TMP
            TMP=GRAD(JK)*T2
            PDOT(K2)=PDOT(K2)-TMP
            PDOT(J2)=PDOT(J2)+TMP
            TMP=GRAD(JK)*T3
            PDOT(K3)=PDOT(K3)-TMP
            PDOT(J3)=PDOT(J3)+TMP
   80    CONTINUE
   90 CONTINUE
C
      RETURN
      END
C
c-----------------------------------------------------------------------
      double precision function ho2gxpot(ider,ra,drv)
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
      r(1) = ra(1)
      r(2) = min(ra(2),ra(3))
      r(3) = max(ra(2),ra(3))
c     if (r(2).ne.ra(2)) print *,'ho2gxpot: echange R2 - R3'
      do 10 i=1,3
   10 r(i) = r(i)/angstrom
      costeta = (r(1)**2+r(2)**2-r(3)**2) / (2.d0*r(1)*r(2))
      gamma = abs(costeta) - 1.d0
c     write(6,105) costeta,gamma
      if (gamma.gt.eps) then
        write(6,103) costeta,r,r(1)+r(2)-r(3)
        stop
      elseif (abs(gamma).lt.eps) then
	costeta = sign(1.d0,costeta)
	linear = .true.
c       print *,'HO2GXPOT: linear =',linear
      end if
      call ho2pes(r(1),r(2),costeta,epot,ider,dvdx)
cxpascho2gxpot = epot*ev - eref
      ho2gxpot = epot - eref/ev + 0.0196725923
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
      if (ra(3).lt.ra(2)) then
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
c    Potential energy surface for the ground electronic state of HO2
c     Please add the following statement in your main program:
c           call potread
c     use the following statement to calculate the PES when needed:
c           call ho2pes(r1,r2,cth,v)
c     where r1 and r2 are the two bondlength O-H in au,
c           th is the enclosed angle in degree
c           v is the potential in hartree 
c-----------------------------------------------------------------------
      subroutine potread
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      parameter (pi=3.141592653589793d0)    
      parameter (noo=50,noh=31,nth=19)
      parameter (m=100,n=100,l=100)
      dimension xt(m),y(m),y2(l),  rrrvec(noo*noh*nth), val(500)  
      logical printgrid
      character*25 grille(2)
      data vmin1/-150.76031165d0/,dy1,dyn/1.0d30,1.0d30/,
     >      printgrid/.false./
      data grille/'ho2pes2007-3-16.txt','ho2pes2007-10-29.txt'/,
     >     igrl/2/
      common /bvcut/vcut
      common /pes/pesmin,rrr(noo,noh,nth),roh(noh),thth(nth),
     &            ve(noo,noh,nth),vs(noo,noh,nth),ind(noh,nth)
c
      write(6,'(''grille de points ab-initio:  '',a25)') grille(igrl)
      open(98,file=grille(igrl))
      vcut=20.0d0/27.212d0
      pesmin=10000
      read(98,*)
      read(98,*)
      do i=1,noh
        do j=1,nth
          do k=1,noo
             read(98,*)ii,ij,ik,ra1,th1,ra2,ve1
             do kk=1,k-1
               if (abs(ra2-rrr(kk,i,j)).lt.0.01) then
	         write(*,*)ra1,th1,ra2
	         stop 'err'
               endif
             enddo
             if (ii+ij+ik.eq.0) goto 11
             roh(i)=ra1
             thth(j)=th1
             rrr(k,i,j)=ra2
             ve(k,i,j)=ve1-vmin1
             if (ve(k,i,j).gt.vcut) ve(k,i,j)=vcut
             if (ve(k,i,j).lt.pesmin) then
               pesmin=ve(k,i,j)
               r1m=ra1
               r2m=ra2
               thm=th1
            endif
          enddo
   11     continue
          kk=k-1
          kkk=kk
          ind(i,j)=kk
        enddo
      enddo
      close(98)
c
      if (printgrid) then
        write(55,*)r1m,r2m,thm
        write(*,*)r1m,r2m,thm,pesmin
        write(*,*)' roh=',(roh(i),i=1,noh)
        write(*,*)' th1=',(thth(i),i=1,nth)
        write(*,*)' nth=',nth,' noh=',noh
        indmin = 100
        indmax = 0
        do i=1,noh
          do j=1,nth
            if (ind(i,j).gt.indmax) indmax = ind(i,j)
	    if (ind(i,j).lt.indmin) indmin = ind(i,j)
            write(55,102) i,j,ind(i,j),rrr(1,i,j),rrr(ind(i,j),i,j)
          enddo
        enddo
        write(55,103)  indmin,indmax
      end if
  102 format(3i7,2f12.6)
  103 format(/'indmin =',i4,', indmax =',i4)
c
      do j=1,nth
        do i=1,noh
          do k=1,ind(i,j)
            ra1=roh(i)
            thjj=thth(j)
            ra2=rrr(k,i,j)
            cthjj=dcos(thjj)
            ra3=dsqrt(ra1*ra1+ra2*ra2-2.0d0*ra1*ra2*cthjj)
c           if (abs(ve(k,i,j)-vcut).lt.1d-4.and.abs(ve(k+1,i,j)).
c     &        lt.vcut)then
c            write(68,1001)thth(j),roh(i),rrr(k,i,j),ve(k,i,j)*27.2114d0
c              write(66,1001)roh(i),thth(j),rrr(k+1,i,j),
c     &        ve(k+1,i,j)*27.2114d0
c           endif 
 1001       format (1x,1f7.3,1f15.8,1f7.3,2f15.8)
          enddo
        enddo
      enddo
c
      ntot = 0
      do i=1,noh
        do j=1,nth
          nh=0
          do k=1,ind(i,j)
	    nh=nh+1
	    ntot = ntot+1
	    rrrvec(ntot) = rrr(k,i,j)
	    xt(nh)=rrr(k,i,j)
	    y(nh)=ve(k,i,j)
          enddo
	  call spline(xt,y,nh,dy1,dyn,y2)
          do ik=1,nh
            vs(ik,i,j)=y2(ik)
          enddo
        enddo
      enddo
c     call listval(ntot,rrrvec,nv1,val,500)
      return
      end
c-----------------------------------------------------------------------
      subroutine ho2pes(r10,r20,cthi,vpot,ider,drv)
c-----------------------------------------------------------------------
c : r1=r(O-O), r2=r(H-O)
      implicit real*8(a-h,o-z)
      parameter (pi=3.141592653589793d0)
      dimension drv(3)
      common /bvcut/vcut
c
      th0=dacos(cthi)*180.0d0/pi
      r1=r10
      r2=r20
      cth=cthi
      r3=dsqrt(r1**2+r2**2-2*r1*r2*cth)
      if (r2.gt.r3) then
	cth=(r1**2+r3**2-r2**2)/(2*r1*r3)
	r2=r3
        print *,'HO2PES: echange R2 - R3'
      endif
      if (r3.lt.0.85) then
        vpot=vcut
	return
      endif
      th=dacos(cth)*180.0d0/pi
      if (r1.lt.1.3) r1=1.3
      if (r2.lt.0.95) r2=0.95
      if (r2.gt.15.0) r2=15.0
c
      call spl3(r1,th,r2,vpot,ider,drv)
c     write(6,100) r1,r2,th,vpot
      if (r2.ge.15.0) then
	drv(2)=0.d0
      end if
      if (r1.ge.20.0) then
	drv(1)=0.d0
      end if
c
      return
  100 format('HO2PES: r1 =',f10.6,', r2 =',f10.6,', theta =',f10.5,
     >      ',Epot =',f12.8)
      end
c-----------------------------------------------------------------------
      subroutine spl3(r1,th,r2,vpot,ider,drv)
c-----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      parameter (noo=50,noh=31,nth=19,m=50)
      dimension ya(m),yw(m),y2a(m),drv(3),dydr1(m),dy2dr1(m),dydth(m),
     >          dyadr1(m),dy2dth(m)
      common /pes/pesmin,rrr(noo,noh,nth),roh(noh),thth(nth),
     >            ve(noo,noh,nth),vs(noo,noh,nth),ind(noh,nth)
      data dy1,dyn/1.0d30,1.0d30/
c
      do 20 i=1,noh
        do 10 j=1,nth
	  nh = ind(i,j)
  	  r1a=r1
          if (r1a.gt.rrr(nh,i,j)) r1a=rrr(nh,i,j)
          call msplint(rrr(1,i,j),ve(1,i,j),vs(1,i,j),nh,r1a,ya(j),
     >                 a,b,c,d,h,klo,khi)
          if (ider.eq.1.and.r1a.le.rrr(nh,i,j)) then
            dd = (1.d0-3.d0*a*a)*vs(klo,i,j)+(3.d0*b*b-1.d0)*vs(khi,i,j)
            dyadr1(j) = (ve(khi,i,j)-ve(klo,i,j))/h + dd*h/6.d0
          else
	    dyadr1(j) = 0.d0
          end if
   10   continue
c
	call spline(thth,ya,nth,0.0d0,0.0d0,y2a)
        call msplint(thth,ya,y2a,nth,th,yw(i),a,b,c,d,h,klo,khi)
        if (ider.eq.1) then
	  call spline(thth,dyadr1,nth,0.0d0,0.0d0,dy2dr1)
          dd = (1.d0-3.d0*a*a)*y2a(klo) + (3.d0*b*b-1.d0)*y2a(khi) 
          dydth(i) = (ya(khi)-ya(klo))/h + dd*h/6.d0
          dydr1(i) = a*dyadr1(klo) + b*dyadr1(khi) + c*dy2dr1(klo) 
     >               + d*dy2dr1(khi)
        end if
   20 continue
c
      call spline(roh,yw,noh,dy1,dyn,y2a)
      call msplint(roh,yw,y2a,noh,r2,vpot,a,b,c,d,h,klo,khi)
      if (ider.eq.1) then
        call spline(roh,dydth,noh,dy1,dyn,dy2dth)
        call spline(roh,dydr1,noh,dy1,dyn,dy2dr1)
        dd = (1.d0-3.d0*a*a)*y2a(klo) + (3.d0*b*b-1.d0)*y2a(khi) 
        drv(2) = (yw(khi)-yw(klo))/h + dd*h/6.d0
        drv(1) = a*dydr1(klo) + b*dydr1(khi) + c*dy2dr1(klo) 
     >           + d*dy2dr1(khi)
        drv(3) = a*dydth(klo) + b*dydth(khi) + c*dy2dth(klo) 
     >           + d*dy2dth(khi)
      end if
c
      return
      end
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
c-----------------------------------------------------------------------
      subroutine spline(x,y,n,yp1,ypn,y2)
c-----------------------------------------------------------------------
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
