      subroutine initialize_potential()
      implicit real*8(a-h,o-z)
      character(len=20) H2Xinput,HHCoeff,XHCoeff,H2XCoeff
      !PARAMETER (N3TMMN = 9)
      !! 
      common /X_prop/X_mass,vdiss,vreac
      common /diatoms/ HHCoeff,XHCoeff,H2XCoeff,H2Xinput


      H2XCoeff='                             '
      HHCoeff ='                             '
      XHCoeff ='                             '
c
             H2Xinput='CNH21App' 
             call Kernel3D(H2Xinput)  
             call property(ventr)
	     write(*,*) 'ventr', ventr
	     write(*,*) 'vdiss', vdiss
	     write(*,*) 'vreac', vreac	    
	     call deriv_coord_type	     

        rAB = 4.05d0  !! N-H  TS
        rBC = 1.42d0  !! H-H' TS
        rAC = 4.05d0  !! N-H' TS

	     
         call PS_H2X(2,rAB,rAC,rBC,VH2N,v2body,v3body)	
	     VH2N=(VH2N)/627.51928d0
         call dPS_H2X(2,rAB,rAC,rBC,dVH2Nx,dVH2Ny,dVH2Nz)	
	     dvH2Nx = dvH2Nx/627.51928d0
	     dvH2Ny = dvH2Ny/627.51928d0
	     dvH2Nz = dvH2Nz/627.51928d0
      end subroutine initialize_potential



      subroutine get_potential(q,Natoms,Nbeads,V,dVdq,info)
      implicit double precision (a-h,o-z)
      integer, intent(in) :: Natoms
      integer, intent(in) :: Nbeads
      double precision, intent(in) :: q(3,Natoms,Nbeads)
      double precision, intent(out) :: V(Nbeads)
      double precision, intent(out) :: dVdq(3,Natoms,Nbeads)

      double precision :: dVdr(1,3), r(1,3)
      double precision :: xAB, yAB, zAB, rAB
      double precision :: xAC, yAC, zAC, rAC
      double precision :: xBC, yBC, zBC, rBC
      integer k, info
      DOUBLE PRECISION :: vH2N
      DOUBLE PRECISION :: v2body,v3body
      DOUBLE PRECISION :: dVH2Nx,dVH2Ny,dVH2Nz     
      character*20 H2Xinput
      DOUBLE PRECISION :: X_mass,vdiss,vreac,ventr  
      common /X_prop/X_mass,vdiss,vreac  

	info = 0 
	v = 0.d0
	dvdq = 0.d0
      do k = 1, Nbeads

	dvdr= 0.d0
	r = 0.d0 

!	rAB
	xAB = q(1,2,k)-q(1,1,k)
	yAB = q(2,2,k)-q(2,1,k)
	zAB = q(3,2,k)-q(3,1,k)
!
!	rAC
	xAC = q(1,3,k)-q(1,1,k)
	yAC = q(2,3,k)-q(2,1,k)
	zAC = q(3,3,k)-q(3,1,k)
!		
!	rBC
	xBC = q(1,3,k)-q(1,2,k)
	yBC = q(2,3,k)-q(2,2,k)
	zBC = q(3,3,k)-q(3,2,k)
!				
	r(1,1) = DSQRT(xAB**2.d0+yAB**2.d0+zAB**2.d0)  ! AB
	rAB = r(1,1)
	r(1,2) = DSQRT(xBC**2.d0+yBC**2.d0+zBC**2.d0)  ! BC
	rBC = r(1,2)
	r(1,3) = DSQRT(xAC**2.d0+yAC**2.d0+zAC**2.d0)  ! AC
	rAC = r(1,3)
!
!
         call PS_H2X(2,rAB,rAC,rBC,VH2N,v2body,v3body)	
         call dPS_H2X(2,rAB,rAC,rBC,dVH2Nx,dVH2Ny,dVH2Nz)	
	     v(k) = (vH2N)/627.51928d0
	     dvdr(1,1) = dVH2Nx/627.51928d0
	     dvdr(1,3) = dVH2Ny/627.51928d0
	     dvdr(1,2) = dVH2Nz/627.51928d0
!
!
!  Atom A
	dvdq(1,1,k) = -dvdr(1,1)*xAB/rAB-dvdr(1,3)*xAC/rAC!       
	dvdq(2,1,k) = -dvdr(1,1)*yAB/rAB-dvdr(1,3)*yAC/rAC!       
	dvdq(3,1,k) = -dvdr(1,1)*zAB/rAB-dvdr(1,3)*zAC/rAC!      
!  Atom B 
	dvdq(1,2,k) = dvdr(1,1)*xAB/rAB-dvdr(1,2)*xBC/rBC!       
	dvdq(2,2,k) = dvdr(1,1)*yAB/rAB-dvdr(1,2)*yBC/rBC!       
	dvdq(3,2,k) = dvdr(1,1)*zAB/rAB-dvdr(1,2)*zBC/rBC!
!  Atom C
	dvdq(1,3,k) = dvdr(1,3)*xAC/rAC+dvdr(1,2)*xBC/rBC!       
	dvdq(2,3,k) = dvdr(1,3)*yAC/rAC+dvdr(1,2)*yBC/rBC!       
	dvdq(3,3,k) = dvdr(1,3)*zAC/rAC+dvdr(1,2)*zBC/rBC!
      
      end do
    
      end subroutine











c
c*********************************************************
c* Illustrations(read carefully before using the program)*
c*********************************************************
c
c       The potential surface and first order derivative of 1A" 
c       states of N+H2 are obtained by calling the subroutines,
c       respectively:
c
c       "PS_H2X(ntype,x,y,z,vH2X,v2body,v3body)" 
c       "dPS_H2X(ntype,x,y,z,dVH2Xx,dVH2Xy,dVH2Xz)"
c
c       ****************************************************************
c
c       Coefficient files: "Coeff.HH-1app", "Coeff.NH-1app","Coeff.NH2-1app"
c       (based on 2715 ab initio data,  Larry Harding, 1999 and 2003)
c 
c       These files contain the parameters for constructing reproducing
c       kernels at the data points(the files "CNH21App"),
c 
c       Distances are in atomic units.
c       Energies are in kcal/mol. (1 a.u.=627.51928 kcal/mol).
c       Angles are in degrees.
c       Potential is taken as zero at the N+H+H limit.. 
c       Vdiss: three-atom dissociation limit (in a.u.)
c       ventr:  The N + H_2 entrance channel energy (in kcal/mol) 
c               w.r.t. the three-atom dissociation limit.
c               Here r_HH = 1.4 a.u.
c
c       The subroutine returns the followings(analytically evaluated):
c 
c          (1) potential "VH2X, v2body, v3body" when "nderiv=0",
c          (2) derivatives "dVH2Xx,y,z"  when "nderiv=1"
c          (3) finite difference  when "nderiv=2"
c
c       at the coordinates (x,y,z) in the 3-D configuration space.
c
c       There are 7 types of coordinate systems supported by the program:
c
c       "ntype" chooses the coordinates for X=O, Ha and Hb.
c       "R_AB" is the distance between the atoms A and B.
c       "R_A-BC" is the distance between the atoms A and the center of mass
c        of the diatom BC.
c
c       ' ntype = 1, ( R_X-H2, R_HaHb, angle )'
c       ' ntype = 2, ( R_XHa, R_XHb, R_HaHb )'
c       ' ntype = 3, ( R_XHa, R_XHb, angle )'
c       ' ntype = 4, ( R_XHa, R_HaHb, angle )'
c       ' ntype = 5, ( x, y, R_HaHb; (x,y) is the position of X with
c                      H-H lying on the x-axis and the origin  at the
c                      center-of-mass of H-H)'
c       ' ntype = 6, ( R_X-HD, R_HD, angle) '
c       ' ntype = 7, ( R_D-XH, R_XH, angle) '
c       ' ntype = 8, ( H(x,y), R_XH) '
c
c
c$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
c****************
c     Example:  *
c****************
c
c     First-order derivatives, dVH2Xx, dVH2Xy, and dVH2Xz,
c     of the H2X potential with respect to
c     x, y, z are calculated analytically.
c
c      The triatomic coordinates x, y, and z are defined by "ntype"
c      as described below.
c
!      implicit real*8(a-h,o-z)
!      character*20 H2Xinput
!      common /X_prop/X_mass,vdiss,vreac
c
c***************************************************************
c    The input files 'CNH21App'contains the file name
c    of expansion coefficients.
c
!           H2Xinput='CNH21App'     ! for 1A" surface
c**************************************************************
c   calling subroutine Kernel3D(H2Xinput) for reading
c   pre-stored expansion coefficients based on ab initio data.
c             *** This is done ONLY ONCE***
c
c         write(6,*) 'Starting presummation!'
!              call Kernel3D(H2Xinput)   ! for 3-body X----H--H
c         write(6,*) 'Presummation done!'
c
!              call property(ventr)
c      ventr : energy for the C(1^D) + H2 channel (w.r.t. 3 atom limit) 
c***************************************************************
!              call deriv_coord_type
c
!      write(6,*) ' need derivatives?'
!      write(6,*) ' nderiv =0, potential only '
!      write(6,*) ' nderiv =1, gradient only '
!      write(6,*) ' nderiv =2, compare analytic and numerical gradient'
!      read (5,*) nderiv
!         if(nderiv.eq.2) then        
c           write(6,*) 'spacing h for finite difference'
!             h=1.d-6
!         endif
!      write(6,*) 'pick a coordinate_type, ntype =1,2,...,8'
!      read (5,*) ntype
!      write(6,*) 'enter the coordinates (x,y,z) now'
!      read (5,*) x,y,z
c
c************************************************************************
c
!      if(nderiv.eq.0) then   !compute potential values only
!          call PS_H2X(ntype,x,y,z,VH2X,v2body,v3body)
!          VH2x=VH2X-ventr  ! zero taken at the C(1^D) + H2 channel
!          write(6,*) x,y,z,vH2x  !au.,au.,au.,kcal/mol
!      endif
c*************************************************************************
!      if(nderiv.eq.1.and.ntype.le.5) then  ! computing gradients
!      call dPS_H2X(ntype,x,y,z,dvH2Xx,dvH2Xy,dvH2Xz)
!      write(6,*) x,y,z,dvH2Xx,dvH2Xy,dvH2Xz !au.,au.,au.,kcal/mol
!      endif
c**********************************************************************
!      if(nderiv.eq.2) then ! finite differencing
c
!              call PS_H2X(ntype,x+h,y,z,V1,v2body,v3body)
!              call PS_H2X(ntype,x-h,y,z,Vm1,v2body,v3body)
!              fdvx=FO3FD(h,v1,vm1)
!              call PS_H2X(ntype,x,y+h,z,V1,v2body,v3body)
!              call PS_H2X(ntype,x,y-h,z,Vm1,v2body,v3body)
!              fdvy=FO3FD(h,v1,vm1)
!              call PS_H2X(ntype,x,y,z+h,V1,v2body,v3body)
!              call PS_H2X(ntype,x,y,z-h,Vm1,v2body,v3body)
!              fdvz=FO3FD(h,v1,vm1)
c
c********************************************************************
c     comparing gradients from analytic and finite difference results
c
!         if(ntype.le.5) then
!             call dPS_H2X(ntype,x,y,z,advx,advy,advz)
!             write(6,*) 'x ',fdvx,advx,fdvx-advx
!             write(6,*) 'y ',fdvy,advy,fdvy-advy
!             write(6,*) 'z ',fdvz,advz,fdvz-advz
!         endif
!      endif
c
! 990  format(2i5)
! 991  format(3e20.8)
!      stop
!      end
c
c
c
      subroutine property(ventr)
      implicit real*8(a-h,o-z)
      x1=100.d0
      y1=1.40d0
      z1=90.d0
      x2=100.d0
      y2=1.97d0
      z2=90.d0
      x3=1.94d0                                             
      y3=x3
      z3=102.7d0
      x4=2.93d0
      y4=1.54d0
      z4=180.d0
      x5=4.0d0
      y5=1.42d0
      z5=90.d0
      call PS_H2X(1,x1,y1,z1,Ventr,v2body,v3body)
      call PS_H2X(7,x2,y2,z2,Vexit,v2body,v3body)
      write(6,'(/)')
      write(6,*) 'exit, entrance, reaction energy = ', Vexit,Ventr,
     &            (Vexit-Ventr), ' kcal/mol'
      call PS_H2X(3,x3,y3,z3,Vmin,v2body,v3body)
      write(6,*) 'minimum energy = ', 
     &            (Vmin-Ventr), ' kcal/mol'
      call PS_H2X(4,x4,y4,z4,Vbari,v2body,v3body)
      write(6,*) 'H--H--N linear barrier position and energy = ', 
     &          x4,y4,z4,(Vbari-Ventr), ' kcal/mol'
      call PS_H2X(1,x5,y5,z5,Vbari,v2body,v3body)
      write(6,*) 'C_2v barrier position and energy = ', 
     &         x5,y5,z5,(Vbari-Ventr), ' kcal/mol'
      write(6,'(/)')
      return
      end
c
c
c     The subroutine deriv_coord_type describes various options
c     for picking coordiante systems and computing gradients
c
      subroutine deriv_coord_type
      implicit real*8(a-h,o-z)
      write(6,*) 'nderiv = 0 : H2X potential only'
      write(6,*) 'nderiv = 1 : 1st-order derivatives only'
      write(6,*) 'nderiv = 2 : derivatives with finite difference'

c
      write(6,*) 'R_AB(or r_AB) is the distance between A and B;'
      write(6,*) 'R_A-BC is the distance between A and the c.m. of BC.' 
c
      write(6,*) 'ntype chooses the coordinates for X(=N), Ha and Hb.'
c
      write(6,*) 'ntype = 1, ( R_X-HH, r_HaHb, angle )'
c
c                             angle between R_X-HH and r_HaHb
c
      write(6,*) 'ntype = 2, ( R_XHa, R_XHb, R_HaHb )'
      write(6,*) 'ntype = 3, ( R_XHa, R_XHb, angle )'
c
c                             angle between R_XHa and R_XHb
c
      write(6,*) 'ntype = 4, ( R_XHa, R_HaHb, angle )'
c
c                             angle between R_XHa and R_HaHb
c
      write(6,*) 'ntype = 5, ( X(x,y), r_HaHb)'
c 
c                x and y defines the position of the atom X in a 2D plane
c                in which the C.M. of the diatom H2 is the origin and
c                lies in the x-direction.                  
c
c     ntype=6,7 for the X(1D) + HD Reaction
c
      write(6,*) 'ntype = 6, ( R_X-HD, r_HD, angle) '
c
c                                 angle between R_X-HD and r_HD
c
      write(6,*) 'ntype = 7, ( R_D-XH, r_XH, angle) '
c
c                                 angle between R_D-XH and r_XH
c
      write(6,*) 'ntype = 8, ( Ha(x,y), r_XHb)'
c 
c                x and y defines the position of the atom Ha in a 2D plane
c                in which the geometric center of the diatom XHb is the origin 
c                and X-Hb lies in the x-direction.   
      return
      end
c
c
c   First-order derivative using three-point finite differnce
c
      function FO3FD(h,f1,fm1)
      implicit real*8(a-h,o-z)
        FO3FD=0.5d0*(f1-fm1)/h
      return
      end
c
c   The subroutine triangle checks the triangular relations between
c   r1, r2 and r3. When the relations are obeyed, return Lflag = 0,
c   otherwise Lflag = 1.
c 
      subroutine triangle(x,y,z,Lflag)
      implicit real*8(a-h,o-z)
      r1=x
      r2=y
      r3=z
      Lflag=0
      if(r1+r2.lt.r3) Lflag=1
      if(r2+r3.lt.r1) Lflag=1
      if(r3+r1.lt.r2) Lflag=1
      return
      end
c
c     The subroutine order rearranges the coordinates for the 2-D contour
c     plots as described in the calling program.
c
      subroutine order(norder,x,y,z,xp,yp,zp)
      implicit real*8(a-h,o-z)
      dimension norder(3)
      dimension rr(3),xx(3)
      rr(1)=x
      rr(2)=y
      rr(3)=z
      do i=1,3
         xx(norder(i))=rr(i)
      enddo
         xp=xx(1)
         yp=xx(2)
         zp=xx(3)
      return
      end
c     
c****************************************
c    The end of the Example!!!!!!!!!    *
c****************************************
c==========================================
c
c    Tak-San Ho 
c    Department of Chemistry
c    Princeton University
c    Princeton, NJ 08544
c    Tel. Phone: (609) 258 6361
c    E-Mail: tsho@princeton.edu
c    Implementation update: Feburary 3, 2003
c
c==========================================

c
c    The subroutine PS_H2X evaluates the H2X potential and its 2-body and
c    3-body terms at the coordinates (x,y,z) as defined by "ntype" in
c    the calling program. 
c
cc     !   VH2X is the H2X potential; VH2X=V2body+V3body;               
cc     !   V2body is the two-body term;                                 
cc     !   V3body is the three-body term.                               
cc     !   The one-body term is absorbed in the two-body term.          
c
      subroutine PS_H2X(ntype,x,y,z,VH2X,v2body,v3body)
      implicit real*8(a-h,o-z)
      parameter (np=30,np1=200)
      dimension x1(np),x2(np),x3(np)
c
      dimension D_H2X(0:np,0:np,0:np,4,4,5)
      common /H2Xfast/ D_H2X
c
      dimension C_HH(0:np1,4)
      common /HHfast/ C_HH
c
      dimension C_XH(0:np1,4)
      common /XHfast/ C_XH
c
      common /X_prop/X_mass,vdiss,vreac
      common /V_prop/a1,a2
      common /p_smooth1/m1,mu1
      common /p_smooth2/m2,mu2
      common /p_smooth3/m3,mu3
      common /coordx1/ x1
      common /coordx2/ x2
      common /coordx3/ x3
      common /n_coordx1/ ns1
      common /n_coordx2/ ns2
      common /n_coordx3/ ns3
c
      pi=dacos(-1.d0)
c
c     computing H2X potential energy surface using many-body-expansion
c
      call xyz_NucDis(ntype,x,y,z,r1,r2,r3)
      call NucDis_sss(r1,r2,r3,s1,s2,s3)
           xout=bond_s1(a1,s1)
           yout=bond_s2(a2,s2)
           zout=bond_s3(s3)
c
         call H2X2bdy(r1,r2,r3,v2body)
         call H2X3bdyQ(xout,yout,zout,V3body)
         VH2X=V2body+V3body 
         VH2X=VH2X*627.51928d0 !converting to kcal/mol
      return
      end
c
c
c     Evaluate the potential gradients
c
      subroutine dPS_H2X(ntype,x,y,zz,dvH2Xx,dvH2Xy,dvH2Xz)
      implicit real*8(a-h,o-z)
      parameter (np=30,np1=200)
      dimension x1(np),x2(np),x3(np)
c
      dimension D_H2X(0:np,0:np,0:np,4,4,5)
      common /H2Xfast/ D_H2X
c
      dimension C_HH(0:np1,4)
      common /HHfast/ C_HH
c
      dimension C_XH(0:np1,4)
      common /XHfast/ C_XH
c
      common /X_prop/X_mass,vdiss,vreac
      common /V_prop/a1,a2
      common /p_smooth1/m1,mu1
      common /p_smooth2/m2,mu2
      common /p_smooth3/m3,mu3
      common /coordx1/ x1
      common /coordx2/ x2
      common /coordx3/ x3
      common /n_coordx1/ ns1
      common /n_coordx2/ ns2
      common /n_coordx3/ ns3
c
      pi=dacos(-1.d0)
      z=zz
c
      call xyz_NucDis(ntype,x,y,z,r1,r2,r3)
      call NucDis_sss(r1,r2,r3,s1,s2,s3)
           xout=bond_s1(a1,s1)
           yout=bond_s2(a2,s2)
           zout=bond_s3(s3)
c
      call dH2X2bdy(r1,r2,r3,dv2bdy1,dV2bdy2,dV2bdy3)
c
      call dH2X3bdyQ(xout,yout,zout,dV3bdyx,dV3bdyy,dV3bdyz)
                 xscale=dbond_s1(a1,s1)
                 yscale=dbond_s2(a2,s2)
                 dV3bdyx=dV3bdyx*xscale
                 dV3bdyy=dV3bdyy*yscale
      call sss_RRR(r1,r2,r3, dv3bdyx,dv3bdyy,dv3bdyz,
     &                       dv3bdy1,dv3bdy2,dv3bdy3)
c
              if(ntype.eq.1) then
                 z=pi*z/180.d0
                 call RRR_Jacobi(x,y,z,r1,r2,r3,
     &                           dv2bdy1,dv2bdy2,dv2bdy3,
     &                           dv2bdyx,dv2bdyy,dv2bdyz)

                 call RRR_Jacobi(x,y,z,r1,r2,r3,
     &                           dv3bdy1,dv3bdy2,dv3bdy3,
     &                           dv3bdyx,dv3bdyy,dv3bdyz)

                 dVH2Xx=dv2bdyx+dv3bdyx
                 dVH2Xy=dv2bdyy+dv3bdyy
                 dVH2Xz=dv2bdyz+dv3bdyz
                 dVH2Xz=dVH2Xz*pi/180.d0
              endif
              if(ntype.ge.2) then
                 dVH2Xx=dv2bdy1+dv3bdy1
                 dVH2Xy=dv2bdy2+dv3bdy2
                 dVH2Xz=dv2bdy3+dv3bdy3
                 if(ntype.eq.3) then
                    z=pi*z/180.d0
                    TJx=(r1-r2*dcos(z))/r3
                    TJy=(r2-r1*dcos(z))/r3
                    TJz=r1*r2*dsin(z)/r3
                    dVH2Xx=dVH2Xx+TJx*dVH2Xz
                    dVH2Xy=dVH2Xy+TJy*dVH2Xz
                    dVH2Xz=dVH2Xz*TJz
                    dVH2Xz=dVH2Xz*pi/180.d0
                 endif
                 if(ntype.eq.4) then
                    z=pi*z/180.d0
                    TJx=(r1-r3*dcos(z))/r2
                    TJy=(r3-r1*dcos(z))/r2
                    TJz=r1*r3/r2*dsin(z)
                    vy=dVH2Xy
                    dVH2Xx=dVH2Xx+TJx*dVH2Xy
                    dVH2Xy=dVH2Xy*TJy+dVH2Xz
                    dVH2Xz=vy*TJz
                    dVH2Xz=dVH2Xz*pi/180.d0
                 endif
                 if(ntype.eq.5) then
                    zhalf=0.5d0*z
                    TJ1x=(x-zhalf)/r1
                    TJ2x=(x+zhalf)/r2
                    TJ1y=y/r1
                    TJ2y=y/r2
                    TJ1z=-0.5d0*TJ1x
                    TJ2z=0.5d0*TJ2x
                    vx=dVH2Xx
                    vy=dVH2Xy
                    dVH2Xx=TJ1x*dVH2Xx+TJ2x*dVH2Xy
                    dVH2Xy=TJ1y*vx+TJ2y*dVH2Xy
                    dVH2Xz=TJ1z*vx+TJ2z*vy+dVH2Xz
                 endif
                 if(ntype.gt.5) then
c
c    derivatives are analytically calculated for ntype =1,2,3,4, and 5 
c
                  write(6,*) '(ntype .gt. 5) is not implemented yet!'
                  stop
                 endif
              endif
              dVH2Xx= dVH2Xx*627.51928d0
              dVH2Xy= dVH2Xy*627.51928d0
              dVH2Xz= dVH2Xz*627.51928d0
      return
      end
c
c     Coordinate transformation from Jacobi ones to internuclear distances
c
      subroutine coord_J(x,y,z,r1,r2,r3)
      implicit real*8(a-h,o-z)
      pi=dacos(-1.d0)
        chi=z*pi/180.d0
        r3=y
        R=x
        cosz=dcos(chi)
        sinz=dsin(chi)
        sinz_sq=sinz*sinz
        r1=dsqrt((R*cosz-0.5d0*y)**2+R*R*sinz_sq)
        r2=dsqrt((R*cosz+0.5d0*y)**2+R*R*sinz_sq)
      return         
      end
c
c     from an arbitrary (x,y,z) to (r1,r2,r3)
c
      subroutine sss_NucDis(s1,s2,s3,r1,r2,r3)
      implicit real*8(a-h,o-z)
      if(s3.eq.0.d0) then
           s3sqrt=0.d0
      else
           s3sqrt=dsqrt(s3)
      endif
      r1=0.5d0*(s1+s2+s2*s3sqrt)
      r2=0.5d0*(s1+s2-s2*s3sqrt)
      r3=s2
      return
      end
c
      subroutine NucDis_sss(r1,r2,r3,s1,s2,s3)
      implicit real*8(a-h,o-z)
      s1=r1+r2-r3
      s2=r3
      if(r3.gt.0.d0) then
         s3=(r1-r2)/r3
         s3=s3*s3
      else
         s3=1.d0
      endif
      return
      end
c
c     from an arbitrary (x,y,z) to (r1,r2,r3)
c
      subroutine xyz_NucDis(ntype,x,y,z,r1,r2,r3)
      implicit real*8(a-h,o-z)
      common /X_prop/X_mass,vdiss,vreac
       pi=dacos(-1.d0)
      if(ntype.eq.1) then
         call coord_J(x,y,z,r1,r2,r3)
      elseif(ntype.eq.2) then
         r1=x
         r2=y
         r3=z
      elseif(ntype.eq.3) then
         r1=x
         r2=y
         chi=pi*z/180.d0
         cosz=dcos(chi)
         sinz=dsin(chi)
         r3=dsqrt((r2-r1*cosz)**2+(r1*sinz)**2)
      elseif(ntype.eq.4) then
         r1=x
         r3=y
            chi=pi*z/180.d0
            cosz=dcos(chi)
            sinz=dsin(chi)
            r2sq=(r1-r3*cosz)**2+(r3*sinz)**2
            r2=dsqrt(r2sq)
      elseif(ntype.eq.5) then
          ysq=y*y
          r3=z
          rhalf=0.5d0*r3
          r1=dsqrt(ysq+(x-rhalf)*(x-rhalf))
          r2=dsqrt(ysq+(x+rhalf)*(x+rhalf))
      elseif(ntype.eq.6) then
          rad=z*pi/180.d0
          a=x*dcos(rad)
          b=x*dsin(rad)
          r3=y
          r_D=1.0078d0/(2.0141d0+1.0078d0)*r3
          r_H=2.0141d0/(2.0141d0+1.0078d0)*r3  
          r1=dsqrt((a-r_D)**2+b*b)
          r2=dsqrt((a+r_H)**2+b*b)
      elseif(ntype.eq.7) then
          r2=y
          r_X=1.0078d0/(X_mass+1.0078d0)*r2
          r_H=X_mass/(X_mass+1.0078d0)*r2   
          rad=z*pi/180.d0
          a=x*dcos(rad)
          b=x*dsin(rad)
          r1=dsqrt((a-r_X)**2+b*b)
          r3=dsqrt((a+r_H)**2+b*b)          
      elseif(ntype.eq.8) then
          r1=z
          r2=dsqrt((x+0.5d0*z)**2+y*y)
          r3=dsqrt((x-0.5d0*z)**2+y*y)
      endif
          if(r1.lt.1.d-16) r1=1.d-16
          if(r2.lt.1.d-16) r2=1.d-16
      return
      end
c
c     Converting coordinates to Jacobi coordinates (R, r, sin(theta))
c
      subroutine xyz_Jacobi(ntype,x,y,z,xOut,yOut,zOut)
      implicit real*8(a-h,o-z)
      pi=dacos(-1.d0)
      if(ntype.eq.1) then
        xout=x
        yout=y
        zout=dsin(z*pi/180.d0)
      elseif(ntype.ge.2.and.ntype.le.7) then
        call  xyz_NucDis(ntype,x,y,z,r1,r2,r3)
c
c       transforming from (r1,r2,r3) to (R,r,sin(theta))
c
        yout=r3
        xout=dsqrt(0.5d0*dabs(r1*r1+r2*r2-0.5d0*r3*r3))
        if(xout.gt.1.d-16.and.r3.gt.1.d-16) then
           zout=0.5d0*(r2*r2-r1*r1)/(r3*xout)
           zout=dsqrt(dabs(1.d0-zout*zout))
        else
           zout=1.d0  ! assume that \theta= 90 degrees when R=0.0 or r=0.0
        endif
      else
        write(6,*) 'ntype.gt.5 is not implemented yet'
        stop
      endif
      return
      end
c
c    potential derivatives
c
      subroutine RRR_Jacobi(x,y,z,r1,r2,r3,
     &                      dv2bdy1,dv2bdy2,dv2bdy3,
     &                      dv2bdyx,dv2bdyy,dv2bdyz)
      implicit real*8(a-h,o-z)
      dimension a(3,3),b(3),c(3)
      b(1)=dv2bdy1
      b(2)=dv2bdy2
      b(3)=dv2bdy3
      sz=dsin(z)
      cz=dcos(z)
      f1=0.5d0*y*cz
      f2=x*cz
      f3=x*y*sz
c
      a(1,1)=(x-f1)/r1
      a(1,2)=(x+f1)/r2
      a(1,3)=0.d0
      a(2,1)=-0.5d0*(f2-0.5d0*y)/r1
      a(2,2)=0.5d0*(f2+0.5d0*y)/r2
      a(2,3)=1.d0
      a(3,1)=0.5d0*f3/r1
      a(3,2)=-0.5d0*f3/r2
      a(3,3)=0.d0
c
      do i=1,3
         c(i)=0.d0
         do j=1,3
            c(i)=c(i)+a(i,j)*b(j)
         enddo
      enddo
      dv2bdyx=c(1)
      dv2bdyy=c(2)
      dv2bdyz=c(3)
      return
      end
c
c     potential derivatives      
c
      subroutine sss_RRR(r1,r2,r3,
     &                      dv3bdyx,dv3bdyy,dv3bdyz,
     &                      dv3bdy1,dv3bdy2,dv3bdy3)
      implicit real*8(a-h,o-z)
      dimension a(3,3),b(3),c(3)
      common /p_smooth3/m3,mu3
      b(1)=dv3bdyx
      b(2)=dv3bdyy
      b(3)=dv3bdyz
c
           a(1,1)=1.d0
           a(2,1)=1.d0
           a(3,1)=-1.d0
c
           a(1,2)=0.d0
           a(2,2)=0.d0
           a(3,2)=1.0
c
           a(1,3)=2.d0*(r1-r2)/r3/r3
           a(2,3)=-a(1,3)
           a(3,3)=-a(1,3)*(r1-r2)/r3
c
      do i=1,3
         c(i)=0.d0
         do j=1,3
            c(i)=c(i)+a(i,j)*b(j)
         enddo
      enddo
      dv3bdy1=c(1)
      dv3bdy2=c(2)
      dv3bdy3=c(3)
      return
      end

c
c     constructing reproducing kernels and their inverses for
c     the variables x1, x2, and x3
c
      subroutine Kernel3D(H2Xinput)
      implicit real*8(a-h,o-z)
      parameter(np=30,np2=np*np,np3=np2*4)
      character*20 H2XCoeff,H2Xinput,HHCoeff,XHCoeff
      dimension icomm(80)
      dimension s1(np),s2(np),s3(np)
      dimension x1(np),x2(np),x3(np)
      dimension Qcff(np,np,np)
      common /Coe3body/ Qcff
c
      common /diatom/ HHCoeff,XHCoeff
      common /X_prop/X_mass,vdiss,vreac
      common /V_prop/a1,a2
      common /p_smooth1/m1,mu1
      common /p_smooth2/m2,mu2
      common /p_smooth3/m3,mu3
c
      common /n_coordx1/ ns1
      common /n_coordx2/ ns2
      common /n_coordx3/ ns3
      common /coordx1/ x1
      common /coordx2/ x2
      common /coordx3/ x3
c
      common /coords1/ s1    !0.leq.s1.< infinity
      common /coords2/ s2    !0.leq.s2.< infinity
      common /coords3/ s3    !0.leq.s3.leq.1

c
c     Read input parameters 
c
      write(6,*) 'reading control file'
      open (unit=10,file=H2Xinput, status='old')
            read (10,992) (icomm(i),i=1,80)
            read (10,*) H2XCoeff   !triatom ab initio data file
            read (10,*) HHCoeff    !diatom H-H ab initio data file
            read (10,*) XHCoeff    !diatom X-H ab initio data file
            read (10,*) X_mass,vdiss,vreac
            read (10,*) m1,m2,m3        !orders of smoothness
            read (10,*) mu1,mu2,mu3     !weights
            read (10,*) a1,a2    !range parameters
            read (10,*) ns1
            do i=1,ns1
               read(10,*) s1(i)
            enddo
            read (10,*) ns2
            do i=1,ns2
               read(10,*) s2(i) 
            enddo
            read (10,*) ns3
            do i=1,ns3
               read(10,*) s3(i)
            enddo
      close (unit=10)
c
      write (6,*) ns1,ns2,ns3
      write(6,*) 'reading 3body coefficients'
      write(6,*) H2XCoeff
         open (unit=11,file=H2XCoeff, status='old')
            do i=1,ns1
               do j=1,ns2
                  do k=1,ns3
                     read (11,*) xs1,xs2,xs3,Qcff(i,j,k)
                  enddo
               enddo
            enddo
         close(unit=11)
c                  do k=1,ns3
c                     write (6,*) s1(1),s2(1),s3(k),Qcff(1,1,k)
c                  enddo
c
      pi=dacos(-1.d0)
c
         write(6,*) HHCoeff,XHCoeff
         call HH2body    ! for 2-body H---H
         call XH2body    ! for 2-body X---H
c
c     converting from bond lenghts to bond-order coordinates
c
         do i=1,ns1
            x1(i)=bond_s1(a1,s1(i))
         enddo
c
         do i=1,ns2
            x2(i)=bond_s2(a2,s2(i))
         enddo
c
          do i=1,ns3
             x3(i)=bond_s3(s3(i))
          enddo

         call QCoeff    ! resummation of 3-body coefficients
         write(6,*) 'Coefficients have been computed successfully!'
c
992   format(80A1)
      return
      end
c
c
c
      double precision function bond_s1(a,s)
      implicit real*8(a-h,o-z)
         bond_s1=dexp(-s*a)
      return
      end
c
      double precision function bond_s2(a,s)
      implicit real*8(a-h,o-z)
         bond_s2=dexp(-s*a)
      return
      end
c
      double precision function bond_s3(s)
      implicit real*8(a-h,o-z)
          bond_s3=s  
      return
      end
c
      double precision function dbond_s1(a,s)
      implicit real*8(a-h,o-z)
         dbond_s1=-a*dexp(-s*a)
      return
      end
c
      double precision function dbond_s2(a,s)
      implicit real*8(a-h,o-z)
         dbond_s2=-a*dexp(-s*a)
      return
      end
c
      double precision function dbond_s3(s)
      implicit real*8(a-h,o-z)
          dbond_s3=1.d0  
      return
      end

c
      subroutine H2X3bdyQ(x,y,z,v3body)
      implicit real*8(a-h,o-z)
      parameter (npp=30)
      dimension x1(npp),x2(npp),x3(npp)
      dimension c_z(0:npp),b(4,4),bz(5)
c
      dimension Bpz(0:50)
      dimension D_H2X(0:npp,0:npp,0:npp,4,4,5)
      common /H2Xfast/ D_H2X
c
      common /p_smooth3/m3,mu3
      common /n_coordx1/ ns1
      common /n_coordx2/ ns2
      common /n_coordx3/ ns3
      common /coordx1/ x1
      common /coordx2/ x2
      common /coordx3/ x3
      call searchB1(ns1,npp,x1,x,ip)
c      write(06,*) ip,x,x1(ip),x1(ip+1)
      call searchB2(ns2,npp,x2,y,jq)
c      write(06,*) jq,y,x2(jq),x1(jq+1)
c    Region I
         b(1,1)=x*y
         b(2,1)=y
         b(3,1)=x
         b(4,1)=1.d0
c    Region II
         b(1,2)=x*y*y
         b(2,2)=y*y
         b(3,2)=x*y*y*y
         b(4,2)=y*y*y
c    Region III
         b(1,3)=x*x*y
         b(2,3)=x*x*x*y
         b(3,3)=x*x
         b(4,3)=x*x*x
c    Region IV
         b(1,4)=x*x*y*y
         b(2,4)=x*x*x*y*y
         b(3,4)=x*x*y*y*y
         b(4,4)=x*x*x*y*y*y
c
c      write(6,*) x,y,z
      call searchB(ns3,npp,x3,z,k)
c      write(06,*) k,z,x3(k),x3(k+1)
c
      bz(2)=z
      bz(3)=1.d0
      bz(4)=z*z
      bz(5)=z*z*z
c
      v3body=0.d0
      do i=1,4
         do j=1,4
               sum = D_H2X(ip,jq,k,j,i,2)*bz(2)
     &              +D_H2X(ip,jq,k,j,i,3)*bz(3)
     &              +D_H2X(ip,jq,k,j,i,4)*bz(4)
     &              +D_H2X(ip,jq,k,j,i,5)*bz(5)
             v3body=v3body+b(j,i)*(sum
     &                      + D_H2X(ip,jq,0,j,i,1)*(1.d0-z)
     &                      + D_H2X(ip,jq,1,j,i,1)*z
     &                      + D_H2X(ip,jq,2,j,i,1)*z*z*(1.d0-z/3.d0)
     &                      + D_H2X(ip,jq,3,j,i,1)*z )
         enddo
      enddo
      v3body=4.d0*v3body
      return
      end
c
      subroutine dH2X3bdyQ(x,y,z,dv3bdyx,dv3bdyy,dv3bdyz)
      implicit real*8(a-h,o-z)
      parameter (npp=30)
      dimension x1(npp),x2(npp),x3(npp)
      dimension c_z(0:npp),b(4,4),bz(5),dbz(5)
c
      dimension Bpz(0:50),dBpz(0:50)
      dimension D_H2X(0:npp,0:npp,0:npp,4,4,5)
      common /H2Xfast/ D_H2X
c
      common /p_smooth3/m3,mu3
      common /n_coordx1/ ns1
      common /n_coordx2/ ns2
      common /n_coordx3/ ns3
      common /coordx1/ x1
      common /coordx2/ x2
      common /coordx3/ x3
c
      call searchB1(ns1,npp,x1,x,ip)
      call searchB2(ns2,npp,x2,y,jq)
      call searchB(ns3,npp,x3,z,k)
c
      zz=z*z
      bz(2)=z
      bz(3)=1.d0
      bz(4)=zz
      bz(5)=zz*z
c
      dbz(2)=1.d0
      dbz(3)=0.d0
      dbz(4)=2.d0*z
      dbz(5)=3.d0*zz
      xx=x*x
      xy=x*y
      yy=y*y
      xxx=xx*x
      xxy=xx*y
      xyy=x*yy
      yyy=yy*y
c
c
c    x-derivative
c
c    Region I
         b(1,1)=y
         b(2,1)=0.d0
         b(3,1)=1.d0
         b(4,1)=0.d0
c    Region II
         b(1,2)=yy
         b(2,2)=0.d0
         b(3,2)=yyy
         b(4,2)=0.d0
c    Region III
         b(1,3)=2.d0*xy
         b(2,3)=3.d0*xxy
         b(3,3)=2.d0*x
         b(4,3)=3.d0*xx
c    Region IV
         b(1,4)=2.d0*xyy
         b(2,4)=3.d0*xx*yy
         b(3,4)=2.d0*x*yyy
         b(4,4)=3.d0*xx*yyy
c
      dV3bdyx=0.d0
      do i=1,4
         do j=1,4
            if(dabs(b(j,i)).gt.0.d0) then
               sum = D_H2X(ip,jq,k,j,i,2)*bz(2)
     &              +D_H2X(ip,jq,k,j,i,3)*bz(3)
     &              +D_H2X(ip,jq,k,j,i,4)*bz(4)
     &              +D_H2X(ip,jq,k,j,i,5)*bz(5)
            dV3bdyx=dV3bdyx+b(j,i)*(sum
     &                      + D_H2X(ip,jq,0,j,i,1)*(1.d0-z)
     &                      + D_H2X(ip,jq,1,j,i,1)*z
     &                      + D_H2X(ip,jq,2,j,i,1)*zz*(1.d0-z/3.d0)
     &                      + D_H2X(ip,jq,3,j,i,1)*z )
           endif
         enddo
      enddo
      dV3bdyx=4.d0*dV3bdyx
c
c    y-derivative
c
c    Region I
         b(1,1)=x
         b(2,1)=1.d0
         b(3,1)=0.d0
         b(4,1)=0.d0
c    Region II
         b(1,2)=2.d0*xy
         b(2,2)=2.d0*y
         b(3,2)=3.d0*xyy
         b(4,2)=3.d0*yy
c    Region III
         b(1,3)=xx
         b(2,3)=xxx
         b(3,3)=0.d0
         b(4,3)=0.d0
c    Region IV
         b(1,4)=2.d0*xxy
         b(2,4)=2.d0*xxx*y
         b(3,4)=3.d0*xxy*y
         b(4,4)=3.d0*xxx*yy
c
      dV3bdyy=0.d0
      do i=1,4
         do j=1,4
            if(dabs(b(j,i)).gt.0.d0) then
               sum = D_H2X(ip,jq,k,j,i,2)*bz(2)
     &              +D_H2X(ip,jq,k,j,i,3)*bz(3)
     &              +D_H2X(ip,jq,k,j,i,4)*bz(4)
     &              +D_H2X(ip,jq,k,j,i,5)*bz(5)
            dV3bdyy= dV3bdyy+b(j,i)*(sum
     &                      + D_H2X(ip,jq,0,j,i,1)*(1.d0-z)
     &                      + D_H2X(ip,jq,1,j,i,1)*z
     &                      + D_H2X(ip,jq,2,j,i,1)*zz*(1.d0-z/3.d0)
     &                      + D_H2X(ip,jq,3,j,i,1)*z )
            endif
         enddo
      enddo
      dV3bdyy=4.d0*dV3bdyy
c
c
c    z-derivative
c
c    Region I
         b(1,1)=xy
         b(2,1)=y
         b(3,1)=x
         b(4,1)=1.d0
c    Region II
         b(1,2)=xyy
         b(2,2)=yy
         b(3,2)=xyy*y
         b(4,2)=yyy
c    Region III
         b(1,3)=xxy
         b(2,3)=xxx*y
         b(3,3)=xx
         b(4,3)=xxx
c    Region IV
         b(1,4)=xxy*y
         b(2,4)=xxx*yy
         b(3,4)=xxy*yy
         b(4,4)=xxx*yyy
c
      dV3bdyz=0.d0
      do i=1,4
         do j=1,4
               sum = D_H2X(ip,jq,k,j,i,2)*dbz(2)
     &              +D_H2X(ip,jq,k,j,i,3)*dbz(3)
     &              +D_H2X(ip,jq,k,j,i,4)*dbz(4)
     &              +D_H2X(ip,jq,k,j,i,5)*dbz(5)
            dV3bdyz=dV3bdyz+b(j,i)*(sum
     &                      - D_H2X(ip,jq,0,j,i,1)
     &                      + D_H2X(ip,jq,1,j,i,1)
     &                      + D_H2X(ip,jq,2,j,i,1)*z*(2.d0-z)
     &                      + D_H2X(ip,jq,3,j,i,1) )
         enddo
      enddo
      dV3bdyz=4.d0*dV3bdyz
c
      return
      end
c   
c
      subroutine Qcoeff
      implicit real*8(a-h,o-z)
      parameter (npp=30)
      dimension Qcff(npp,npp,npp)
      dimension x1(npp),x2(npp),x3(npp)
c
      dimension C_H2X(0:npp,0:npp,npp,4,4)
      dimension D_H2X(0:npp,0:npp,0:npp,4,4,5)   !,B_p(npp,0:50)
      common /H2Xfast/ D_H2X
      common /Coe3body/ Qcff
c
      common /p_smooth3/m3,mu3
      common /n_coordx1/ ns1
      common /n_coordx2/ ns2
      common /n_coordx3/ ns3
      common /coordx1/ x1
      common /coordx2/ x2
      common /coordx3/ x3
c
c    presum coefficients related to the ab initio data points
c
c    Note: the magnitudes of x1 and x2 decrease as the indices i, j increase.
c
      do k=1,ns3
c          Region I
        do ip=0,ns1
        do jq=0,ns2
           C_H2X(ip,jq,k,1,1)=0.d0
           C_H2X(ip,jq,k,2,1)=0.d0
           C_H2X(ip,jq,k,3,1)=0.d0
           C_H2X(ip,jq,k,4,1)=0.d0
           if(ip.lt.ns1.and.jq.lt.ns2) then
             do i=ip+1,ns1
             do j=jq+1,ns2
                a1=x1(i)*x1(i)*x2(j)*x2(j)
                a2=a1*x1(i)
                a3=a1*x2(j)
                a4=a2*x2(j)
                C_H2X(ip,jq,k,1,1)=C_H2X(ip,jq,k,1,1)+Qcff(i,j,k)*a1
                C_H2X(ip,jq,k,2,1)=C_H2X(ip,jq,k,2,1)+Qcff(i,j,k)*a2
                C_H2X(ip,jq,k,3,1)=C_H2X(ip,jq,k,3,1)+Qcff(i,j,k)*a3
                C_H2X(ip,jq,k,4,1)=C_H2X(ip,jq,k,4,1)+Qcff(i,j,k)*a4
             enddo
             enddo
                C_H2X(ip,jq,k,2,1)=-C_H2X(ip,jq,k,2,1)/3.d0
                C_H2X(ip,jq,k,3,1)=-C_H2X(ip,jq,k,3,1)/3.d0
                C_H2X(ip,jq,k,4,1)= C_H2X(ip,jq,k,4,1)/9.d0
          endif

        enddo
        enddo
c           Region II
        do ip=0,ns1
        do jq=0,ns2
           C_H2X(ip,jq,k,1,2)=0.d0
           C_H2X(ip,jq,k,2,2)=0.d0
           C_H2X(ip,jq,k,3,2)=0.d0
           C_H2X(ip,jq,k,4,2)=0.d0
           if(ip.lt.ns1.and.jq.gt.0) then
             do i=ip+1,ns1
             do j=1,jq
                a=x1(i)*x2(j)
                a1=a*x1(i)
                a2=a1*x1(i)
                a3=x1(i)*x1(i)
                a4=a3*x1(i)
                C_H2X(ip,jq,k,1,2)=C_H2X(ip,jq,k,1,2)+Qcff(i,j,k)*a1
                C_H2X(ip,jq,k,2,2)=C_H2X(ip,jq,k,2,2)+Qcff(i,j,k)*a2
                C_H2X(ip,jq,k,3,2)=C_H2X(ip,jq,k,3,2)+Qcff(i,j,k)*a3
                C_H2X(ip,jq,k,4,2)=C_H2X(ip,jq,k,4,2)+Qcff(i,j,k)*a4
             enddo
             enddo
                C_H2X(ip,jq,k,2,2)=-C_H2X(ip,jq,k,2,2)/3.d0
                C_H2X(ip,jq,k,3,2)=-C_H2X(ip,jq,k,3,2)/3.d0
                C_H2X(ip,jq,k,4,2)= C_H2X(ip,jq,k,4,2)/9.d0
          endif

        enddo
        enddo
c          Region III

        do ip=0,ns1
        do jq=0,ns2
           C_H2X(ip,jq,k,1,3)=0.d0
           C_H2X(ip,jq,k,2,3)=0.d0
           C_H2X(ip,jq,k,3,3)=0.d0
           C_H2X(ip,jq,k,4,3)=0.d0
           if(ip.gt.0.and.jq.lt.ns2) then
             do i=1,ip
             do j=jq+1,ns2
                a=x1(i)*x2(j)
                a1=a*x2(j)
                a2=x2(j)*x2(j)
                a3=a1*x2(j)
                a4=a2*x2(j)
                C_H2X(ip,jq,k,1,3)=C_H2X(ip,jq,k,1,3)+Qcff(i,j,k)*a1
                C_H2X(ip,jq,k,2,3)=C_H2X(ip,jq,k,2,3)+Qcff(i,j,k)*a2
                C_H2X(ip,jq,k,3,3)=C_H2X(ip,jq,k,3,3)+Qcff(i,j,k)*a3
                C_H2X(ip,jq,k,4,3)=C_H2X(ip,jq,k,4,3)+Qcff(i,j,k)*a4
             enddo
             enddo
                C_H2X(ip,jq,k,2,3)=-C_H2X(ip,jq,k,2,3)/3.d0
                C_H2X(ip,jq,k,3,3)=-C_H2X(ip,jq,k,3,3)/3.d0
                C_H2X(ip,jq,k,4,3)= C_H2X(ip,jq,k,4,3)/9.d0
          endif

        enddo
        enddo
c          Region IV
        do ip=0,ns1
        do jq=0,ns2
           C_H2X(ip,jq,k,1,4)=0.d0
           C_H2X(ip,jq,k,2,4)=0.d0
           C_H2X(ip,jq,k,3,4)=0.d0
           C_H2X(ip,jq,k,4,4)=0.d0
           if(ip.gt.0.and.jq.gt.0) then
             do i=1,ip
             do j=1,jq
                a1=x1(i)*x2(j)
                a2=x2(j)
                a3=x1(i)
                C_H2X(ip,jq,k,1,4)=C_H2X(ip,jq,k,1,4)+Qcff(i,j,k)*a1
                C_H2X(ip,jq,k,2,4)=C_H2X(ip,jq,k,2,4)+Qcff(i,j,k)*a2
                C_H2X(ip,jq,k,3,4)=C_H2X(ip,jq,k,3,4)+Qcff(i,j,k)*a3
                C_H2X(ip,jq,k,4,4)=C_H2X(ip,jq,k,4,4)+Qcff(i,j,k)
             enddo
             enddo
                C_H2X(ip,jq,k,2,4)=-C_H2X(ip,jq,k,2,4)/3.d0
                C_H2X(ip,jq,k,3,4)=-C_H2X(ip,jq,k,3,4)/3.d0
                C_H2X(ip,jq,k,4,4)= C_H2X(ip,jq,k,4,4)/9.d0
          endif

        enddo
        enddo
      enddo

c     presum over data in the z-coordinate
c
      do i=1,4
      do j=1,4
      do jq=0,ns2
      do ip=0,ns1
            sum01=0.d0
            sum11=0.d0
            sum21=0.d0
            sum31=0.d0
         do kk=1,ns3
            sum01=sum01+C_H2X(ip,jq,kk,j,i)*(1-x3(kk))
            sum11=sum11+C_H2X(ip,jq,kk,j,i)*x3(kk)*7.d0/3.d0
            sum21=sum21-C_H2X(ip,jq,kk,j,i)*x3(kk)*2.d0
            sum31=sum31-C_H2X(ip,jq,kk,j,i)*2.d0*x3(kk)*x3(kk)
     &                                      *(1.d0-x3(kk)/3.d0)
         enddo
            D_H2X(ip,jq,0,j,i,1)=sum01
            D_H2X(ip,jq,1,j,i,1)=sum11
            D_H2X(ip,jq,2,j,i,1)=sum21
            D_H2X(ip,jq,3,j,i,1)=sum31
c
      do k=0,ns3
            sum2=0.d0
            sum3=0.d0
         do kk=1,k
            sum2=sum2+C_H2X(ip,jq,kk,j,i)*2.d0*pow(2,x3(kk))
            sum3=sum3-C_H2X(ip,jq,kk,j,i)*2.d0*pow(3,x3(kk))/3.d0
         enddo
            D_H2X(ip,jq,k,j,i,2)=sum2
            D_H2X(ip,jq,k,j,i,3)=sum3
c
            sum4=0.d0
            sum5=0.d0
         do kk=k+1,ns3
            sum4=sum4+C_H2X(ip,jq,kk,j,i)*2.d0*x3(kk)
            sum5=sum5-C_H2X(ip,jq,kk,j,i)*2.d0/3.d0
         enddo
            D_H2X(ip,jq,k,j,i,4)=sum4
            D_H2X(ip,jq,k,j,i,5)=sum5
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
c
      subroutine H2X2bdy(r1,r2,r3,V2bdy)
      implicit real*8(a-h,o-z)
      vXH1=v2XH(r1)
      vXH2=v2XH(r2)
      vhh=v2hh(r3)
      V2bdy=vXH1+vXH2+vhh 
      return
      end
c
      subroutine dH2X2bdy(r1,r2,r3,dV2bdy1,dV2bdy2,dV2bdy3)
      implicit real*8(a-h,o-z)
      dv2bdy1=dv2XH(r1)
      dv2bdy2=dv2XH(r2)
      dv2bdy3=dv2hh(r3)
      return
      end
c
c     constructing two-body H-H intermolecular potential energy surfaces
c
      subroutine HH2body
      implicit real*8(a-h,o-z)
      parameter(np=200)
      character*20 HHCoeff,XHCoeff
      dimension icomm(80)
      dimension r1(np)
      dimension x1(np)
      dimension qc(np)
c
      dimension C_HH(0:np,4)
      common /HHfast/ C_HH
c
      common /diatom/ HHCoeff,XHCoeff
      common /HH2b1/ VdHH,ReHH
      common /HH2b2/ m1,mu,nr1
      common /HH2b4/ r1,x1

      open (unit=11,file=HHCoeff, status='old')
            read (11,992) (icomm(i),i=1,80)
            read (11,*) VdHH
            read (11,*) m1,mu        !orders of smoothness
            read (11,*) ReHH
            read (11,*) nr1
            do i=1,nr1
               read (11,*) r1(i),qc(i)
            enddo
      close (unit=11)
992   format(80A1)
c             write(6,*) nr1
c            do i=1,nr1
c               write (6,*) r1(i),qc(i)
c            enddo
c
c     converting from bond lenghts to bond-order coordinates
         do i=1,nr1
            x1(i)=r1(i)
         enddo
c
c    presum coefficients
c
      n=nr1
      do k=0,n
         C_HH(k,1)=0.d0
         C_HH(k,2)=0.d0
         if(k.gt.0) then
            do i=1,k
               C_HH(k,1)=C_HH(k,1)+qc(i)
               C_HH(k,2)=C_HH(k,2)+qc(i)*x1(i)
            enddo
         endif
         C_HH(k,1)=4.d0/dfloat((mu+1)*(mu+2))*C_HH(k,1)
         C_HH(k,2)=-4.d0/dfloat((mu+2)*(mu+3))*C_HH(k,2)
c         C_HH(k,1)=2.d0*C_HH(k,1)
c         C_HH(k,2)=-2.d0/3.d0*C_HH(k,2)
      enddo
c
      do k=0,n
         C_HH(k,3)=0.d0
         C_HH(k,4)=0.d0
         if(k.lt.n) then 
            do i=k+1,n
               C_HH(k,3)=C_HH(k,3)+qc(i)/pow(mu+1,x1(i))
               C_HH(k,4)=C_HH(k,4)+qc(i)/pow(mu+2,x1(i))
c               C_HH(k,3)=C_HH(k,3)+qc(i)/x1(i)
c               C_HH(k,4)=C_HH(k,4)+qc(i)/( x1(i)*x1(i) )
            enddo
         endif
         C_HH(k,3)=4.d0/dfloat((mu+1)*(mu+2))*C_HH(k,3)
         C_HH(k,4)=-4.d0/dfloat((mu+2)*(mu+3))*C_HH(k,4)
c         C_HH(k,3)=2.d0*C_HH(k,3)
c         C_HH(k,4)=-2.d0/3.d0*C_HH(k,4)
      enddo         
c
      return
      end
c
      function V2HH(rr1)
      implicit real*8(a-h,o-z)
      parameter(np=200)
      dimension r1(np)
      dimension x1(np)
c
      dimension C_HH(0:np,4)
      common /HHfast/ C_HH
c
      common /HH2b1/ VdHH,ReHH
      common /HH2b2/ m1,mu,nr1
      common /HH2b4/ r1,x1
c
         n=nr1
         xx1=rr1
         call searchB(n,np,x1,xx1,k)
ccc         write(06,*) k,xx1,x1(k),x1(k+1)
         if(k.eq.0) then
            V2HH=C_HH(k,3)+C_HH(k,4)*xx1
         elseif(k.eq.n) then
            a1=1.d0/pow(mu+1,xx1)
            a2=a1/xx1
c            a1=1.d0/xx1
c            a2=a1*a1
            V2HH=C_HH(k,1)*a1+C_HH(k,2)*a2
         else
            a1=1.d0/pow(mu+1,xx1)
            a2=a1/xx1
c            a1=1.d0/xx1
c            a2=a1*a1
            V2HH=C_HH(k,1)*a1+C_HH(k,2)*a2+C_HH(k,3)+C_HH(k,4)*xx1
         endif
         V2HH=V2HH   !+VdHH
      return
      end
c
      function dV2HH(rr1)
      implicit real*8(a-h,o-z)
      parameter(np=200)
      dimension r1(np)
      dimension x1(np)
c
      dimension C_HH(0:np,4)
      common /HHfast/ C_HH
c
      common /HH2b1/ VdHH,ReHH
      common /HH2b2/ m1,mu,nr1
      common /HH2b4/ r1,x1
c
         n=nr1
         xx1=rr1
         call searchB(n,np,x1,xx1,k)
         if(k.eq.0) then 
            dV2HH=C_HH(k,4)
         elseif(k.eq.n) then
            a=1.d0/pow(mu+2,xx1)
            a1=-a*dfloat(mu+1)
            a2=-a/xx1*dfloat(mu+2)
c            a=1.d0/xx1/xx1
c            a1=-1.d0*a
c            a2=-2.d0*a/xx1
            dV2HH=C_HH(k,1)*a1+C_HH(k,2)*a2
         else
            a=1.d0/pow(mu+2,xx1)
            a1=-a*dfloat(mu+1)
            a2=-a/xx1*dfloat(mu+2)
c            a=1.d0/xx1/xx1
c            a1=-1.d0*a
c            a2=-2.d0*a/xx1
            dV2HH=C_HH(k,1)*a1+C_HH(k,2)*a2+C_HH(k,4)
         endif
      return
      end
c
c     constructing two-body X-H intermolecular potential energy surfaces
c
      subroutine XH2body
      implicit real*8(a-h,o-z)
      parameter(np=200)
      character*20 HHCoeff,XHCoeff
      dimension icomm(80)
      dimension r1(np)
      dimension x1(np)
      dimension qc(np)
c
      dimension C_XH(0:np,4)
      common /XHfast/ C_XH
c
      common /diatom/ HHCoeff,XHCoeff
      common /XH2b1/ VdXH,ReXH
      common /XH2b2/ m1,mu,nr1
      common /XH2b4/ r1,x1
c
      open (unit=11,file=XHCoeff, status='old')
            read (11,992) (icomm(i),i=1,80)
            read (11,*) VdXH
            read (11,*) m1,mu        !orders of smoothness
            read (11,*) ReXH
            read (11,*) nr1
            do i=1,nr1
               read (11,*) r1(i),qc(i)
            enddo
      close (unit=11)
992   format(80A1)
c            do i=1,nr1
c               write (6,*) r1(i),qc(i)
c            enddo
c
c     converting from bond lenghts to bond-order coordinates
         do i=1,nr1
            x1(i)=r1(i)
         enddo
c
c    presum coefficients
c
      n=nr1
      do k=0,n
         C_XH(k,1)=0.d0
         C_XH(k,2)=0.d0
         if(k.gt.0) then
            do i=1,k
               C_XH(k,1)=C_XH(k,1)+qc(i)
               C_XH(k,2)=C_XH(k,2)+qc(i)*x1(i)
            enddo
         endif
         C_XH(k,1)=4.d0/dfloat((mu+1)*(mu+2))*C_XH(k,1)
         C_XH(k,2)=-4.d0/dfloat((mu+2)*(mu+3))*C_XH(k,2)
c         C_XH(k,1)=2.d0*C_XH(k,1)
c         C_XH(k,2)=-2.d0/3.d0*C_XH(k,2)
      enddo
c
      do k=0,n
         C_XH(k,3)=0.d0
         C_XH(k,4)=0.d0
         if(k.lt.n) then 
            do i=k+1,n
               C_XH(k,3)=C_XH(k,3)+qc(i)/pow(mu+1,x1(i))
               C_XH(k,4)=C_XH(k,4)+qc(i)/pow(mu+2,x1(i))
c               C_XH(k,3)=C_XH(k,3)+qc(i)/x1(i)
c               C_XH(k,4)=C_XH(k,4)+qc(i)/( x1(i)*x1(i) )
            enddo
         endif
         C_XH(k,3)=4.d0/dfloat((mu+1)*(mu+2))*C_XH(k,3)
         C_XH(k,4)=-4.d0/dfloat((mu+2)*(mu+3))*C_XH(k,4)
c         C_XH(k,3)=2.d0*C_XH(k,3)
c         C_XH(k,4)=-2.d0/3.d0*C_XH(k,4)
      enddo         
      return
      end
c
      function V2XH(rr1)
      implicit real*8(a-h,o-z)
      parameter(np=200)
      dimension r1(np)
      dimension x1(np)
c
      dimension C_XH(0:np,4)
      common /XHfast/ C_XH
c
      common /XH2b1/ VdXH,ReXH
      common /XH2b2/ m1,mu,nr1
      common /XH2b4/ r1,x1
c
c     Constructing bases
c
         n=nr1
         xx1=rr1
         call searchB(n,np,x1,xx1,k)
ccc         write(06,*) k,xx1,x1(k),x1(k+1)
         if(k.eq.0) then
            V2XH=C_XH(k,3)+C_XH(k,4)*xx1
         elseif(k.eq.n) then
            a1=1.d0/pow(mu+1,xx1)
            a2=a1/xx1
c            a1=1.d0/xx1
c            a2=a1*a1
            V2XH=C_XH(k,1)*a1+C_XH(k,2)*a2
         else
            a1=1.d0/pow(mu+1,xx1)
            a2=a1/xx1
c            a1=1.d0/xx1
c            a2=a1*a1
            V2XH=C_XH(k,1)*a1+C_XH(k,2)*a2+C_XH(k,3)+C_XH(k,4)*xx1
         endif
         V2XH=V2XH   !+VdXH
      return
      end
c
      function dV2XH(rr1)
      implicit real*8(a-h,o-z)
      parameter(np=200)
      dimension r1(np)
      dimension x1(np)
c
      dimension C_XH(0:np,4)
      common /XHfast/ C_XH
c
      common /XH2b1/ VdXH,ReXH
      common /XH2b2/ m1,mu,nr1
      common /XH2b4/ r1,x1
c
         n=nr1
         xx1=rr1
         call searchB(n,np,x1,xx1,k)
         if(k.eq.0) then 
            dV2XH=C_XH(k,4)
         elseif(k.eq.n) then
            a=1.d0/pow(mu+2,xx1)
            a1=-a*dfloat(mu+1)
            a2=-a/xx1*dfloat(mu+2)
c            a=1.d0/xx1/xx1
c            a1=-1.d0*a
c            a2=-2.d0*a/xx1
            dV2XH=C_XH(k,1)*a1+C_XH(k,2)*a2
         else
            a=1.d0/pow(mu+2,xx1)
            a1=-a*dfloat(mu+1)
            a2=-a/xx1*dfloat(mu+2)
c            a=1.d0/xx1/xx1
c            a1=-1.d0*a
c            a2=-2.d0*a/xx1
            dV2XH=C_XH(k,1)*a1+C_XH(k,2)*a2+C_XH(k,4)
         endif
      return
      end
c
c     Binary search 
c
      subroutine searchB(n,np,xx,x,k)
      implicit real*8(a-h,o-z)
      dimension xx(np)
      if(x.lt.xx(1)) then
         k=0
      elseif(x.ge.xx(n)) then
         k=n
      else
         i=1
         j=n
 999  continue
         kk=(i+j)/2
         if(x.lt.xx(kk)) then
               j=kk
         else
               i=kk
         endif
         if(x.ge.xx(kk).and.x.lt.xx(kk+1)) then
               k=kk
         else
               goto 999
         endif
      endif
      return
      end
c
c     Binary search 
c
      subroutine searchB1(n,np,xx,x,k)
      implicit real*8(a-h,o-z)
      dimension xx(np)
      if(x.gt.xx(1)) then
         k=0
      elseif(x.le.xx(n)) then
         k=n
      else
         i=1
         j=n
 999  continue
         kk=(i+j)/2
         if(x.gt.xx(kk)) then
               j=kk
         else
               i=kk
         endif
         if(x.le.xx(kk).and.x.gt.xx(kk+1)) then
               k=kk
         else
               goto 999
         endif
      endif
      return
      end
c
      subroutine searchB2(n,np,xx,x,k)
      implicit real*8(a-h,o-z)
      dimension xx(np)
      if(x.gt.xx(1)) then
         k=0
      elseif(x.le.xx(n)) then
         k=n
      else
         i=1
         j=n
 999  continue
         kk=(i+j)/2
         if(x.gt.xx(kk)) then
               j=kk
         else
               i=kk
         endif
         if(x.le.xx(kk).and.x.gt.xx(kk+1)) then
               k=kk
         else
               goto 999
         endif
      endif
      return
      end

c
      function pow(m,x)
      implicit real*8(a-h,o-z)
      if(m.eq.0) then
        pow=1.
      elseif(m.gt.0) then
        pow=1.
        do i=1,m
           pow=pow*x
        enddo
      elseif(m.lt.0) then
        pow=1.
        do i=1,-m
           pow=pow*x
        enddo
           pow=1./pow
      endif
      return
      end

