!=============================================================
      module mainvar
!=============================================================
      implicit none

!     Constants:
      complex*16, parameter :: jcpx=(0.d0,1.d0) !imaginary unit
      real*8, parameter    :: pi = 4.d0*atan(1.d0)

!     Variables:
      integer :: nstate,nstep,ngrid,nwrite,istate_ini
      real*8  :: xmin,xmax
      real*8  :: dt,dx
      real*8  :: redmass,k_ini,sigma_ini,xcenter_ini
      real*8, allocatable :: xgrid(:)
      real*8, allocatable :: Cpop(:,:),Cpop_ad(:,:)
      complex*16, allocatable :: psi(:,:), Vadiabatic(:,:)
      complex*16, allocatable :: psi_ad(:,:)
      complex*16, allocatable :: Op_kin(:)
      complex*16, allocatable :: Op_pot(:,:,:), DtoA(:,:,:)


      
      contains
     !--------------------------------------------
      subroutine allocate_main_arrays(nalloc)
      implicit none
      integer :: nalloc

      if(nalloc == 1) then
         allocate(xgrid(ngrid))
         allocate(Cpop(nstate,0:nstep))
         allocate(Cpop_ad(nstate,0:nstep))
         allocate(psi(nstate,ngrid))
         allocate(psi_ad(nstate,ngrid))
         allocate(Vadiabatic(nstate,ngrid))
         allocate(Op_pot(nstate,nstate,ngrid))
         allocate(DtoA(nstate,nstate,ngrid))
         allocate(Op_kin(ngrid))
      else
         deallocate(xgrid)
         deallocate(Vadiabatic)
         deallocate(DtoA)
         deallocate(psi)
         deallocate(psi_ad)
         deallocate(Op_pot)
         deallocate(Op_kin)
         deallocate(Cpop)
         deallocate(Cpop_ad)
      endif 

      end subroutine
     !--------------------------------------------

      endmodule
!=============================================================


      program wp_propag
      use mainvar
      implicit none
      integer :: i,iE,nE
      real*8, allocatable :: k_ini_array(:)
      character(len=20) :: str

     !Read input
      read(*,*) nstate
      read(*,*) ngrid,xmin,xmax
      read(*,*) nstep, dt, nwrite
      read(*,*) nE,istate_ini,xcenter_ini,redmass

      allocate(k_ini_array(nE))

      read(*,*) (k_ini_array(i),i=1,nE)
      
     !grid resolution
      dx=(xmax-xmin)/dble(ngrid-1)


     !Allocate variables
      call allocate_main_arrays(1)

     !---------------------------------
      do iE=1,nE
         k_ini=k_ini_array(iE)

         open(65,file='WF_traj_diabatic_'//trim(str(iE))//'.dat')
         open(66,file='WF_traj_adiabatic_'//trim(str(iE))//'.dat')
         open(66,file='Population_diabatic_'//trim(str(iE))//'.dat')
         open(68,file='Population_adiabatic_'//trim(str(iE))//'.dat')

         !Initialize the Gaussian WP
         call Init_Psi

         !Build the Pot and Kin energy part of the propagator
         call Set_Operators

         !Print the initial WP
         call write_psi(0)

         !Save Initial state population
         call Save_State_Population(0)

         !Build the transformation matrix between the two representation
         call Build_Diabatic_to_Adiabatic_matrix

         call Do_Propagation

         close(65)
         close(66)
         close(67)
         close(68)
      enddo
     !---------------------------------
    

     !Deallocate the variables
      call allocate_main_arrays(0)

      end program wp_propag


!======================================================================================
      character(len=20) function str(k)
!======================================================================================
!      Convert an integer to string  
!======================================================================================
      integer, intent(in) :: k
      write (str, *) k
      str = adjustl(str)
      end function str
!======================================================================================


!======================================================================================
      subroutine Do_Propagation
!======================================================================================
!     FFT time-propagation with symmetric Trotter-splitting
!======================================================================================
      use mainvar
      implicit none
      integer :: i 

      do i=1,nstep

         call Propagate_One_Step

        !Print wavefunction
         if(mod(i,nwrite) == 0) then
             call write_psi(i)
             write(*,*) "time step:",i
         endif
         
         call Save_State_Population(i)

         call print_checkpoint_psi(i)

      enddo
      end subroutine
!======================================================================================




!======================================================================================
      subroutine Save_State_Population(istep)
!======================================================================================
      use mainvar
      implicit none
      integer, save  :: icall_first=1
      integer        :: ig,is,istep
      real*8         :: norm,norm_ad

      do is=1,nstate

         norm=0.0d0
         norm_ad=0.0d0
         do ig=1,ngrid
            norm = norm + dx*psi(is,ig)*conjg(psi(is,ig))
            norm_ad = norm_ad + dx*psi_ad(is,ig)*conjg(psi_ad(is,ig))
         enddo

         Cpop(is,istep)=norm
         Cpop_ad(is,istep)=norm_ad
      enddo


      if(icall_first == 1) then
         write(67,*) k_ini,"k_ini"
         write(68,*) k_ini,"k_ini"
         icall_first=99
      else

        write(67,"(f12.5,6e15.5)") istep*dt,(Cpop(is,istep),is=1,nstate)
        write(68,"(f12.5,6e15.5)") istep*dt,(Cpop_ad(is,istep),is=1,nstate)
      endif





      end subroutine
!======================================================================================


!======================================================================================
      subroutine Vpot(x,V)
!======================================================================================
!     This function has to be changed to simulate a given system
!======================================================================================
      use mainvar
      implicit none
      real*8 :: x
      real*8, parameter :: A=0.01, B=1.6, C=0.005, D=1.0
      real*8, dimension(nstate,nstate) :: V

!     Fewest surface hopping algorithm as introduced in the article
!     Tully, J.C. J. Chem. Phys. (1990) 93 1061. 

      V = 0.0
      if(x>0) then
          V(1,1) = A*(1.0-exp(-B*x))
      else
          V(1,1) = -A*(1.0-exp(B*x))
      endif
      V(2,2) = -V(1,1)
      V(1,2) = C*exp(-D*x*x)
      V(2,1) = V(1,2)

      end subroutine
!======================================================================================




!======================================================================================
      subroutine print_real_matrix(ifile,it,n,A)
!======================================================================================
      implicit none
      integer :: n,ifile,j,i,it
      real*8, dimension(n,n) :: A


      write(ifile,*) "now printing:",it
      do i=1,n
         write(ifile,"(10e12.4)") (A(i,j),j=1,n)
      enddo
      call flush(ifile)

      end subroutine
!======================================================================================


!======================================================================================
      subroutine Propagate_One_Step
!======================================================================================
!     FFT time-propagation with symmetric Trotter-splitting              
!     Only one time step is done in this routine
!======================================================================================
      use mainvar
      implicit none
      integer :: ig,is
      complex*16, dimension(nstate,nstate) :: Vc
      complex*16, dimension(ngrid) :: phi
      complex*16, dimension(nstate) :: chi

     !----------------------------------------------
     !Half-potential step: exp(-i*V/2)*Psi(x)
     !----------------------------------------------
      do ig=1,ngrid
         Vc=Op_pot(:,:,ig)
         chi=psi(:,ig)

         chi=matmul(Vc,chi)

         psi(:,ig)=chi
      enddo
     !----------------------------------------------


     !----------------------------------------------
     !FFT forward: Psi(x) --> Psi(k)
     !----------------------------------------------
      do is=1,nstate
         phi=psi(is,:)
         call fourier(0,ngrid,phi) !initialize
         call fourier(1,ngrid,phi) !do forwad FFT
         call fourier(2,ngrid,phi) !deallocate

         psi(is,:)=phi
      enddo
     !----------------------------------------------


     !----------------------------------------------
     !Kinetic energy step: exp(-i*T)*Psi(k)
     !----------------------------------------------
      do is=1,nstate
         phi=psi(is,:)

         phi=Op_Kin*phi

         psi(is,:)=phi
      enddo
     !----------------------------------------------


     !----------------------------------------------
     !FFT backward: Psi(k) --> Psi(x)
     !----------------------------------------------
      do is=1,nstate
         phi=psi(is,:)
         call fourier(0,ngrid,phi) !initilize
         call fourier(-1,ngrid,phi)!do inverse FFT
         call fourier(2,ngrid,phi) !deallocate

         psi(is,:)=phi
      enddo
     !----------------------------------------------


     !----------------------------------------------
     !Half-potential step: exp(-i*V/2)*Psi(x)
     !----------------------------------------------
      do ig=1,ngrid
         Vc=Op_pot(:,:,ig)
         chi=psi(:,ig)

         chi=matmul(Vc,chi)

         psi(:,ig)=chi
      enddo
     !----------------------------------------------


     !Normalize the wavefunction
      call normalize_psi


      do ig=1,ngrid
         psi_ad(:,ig)=matmul(transpose(DtoA(:,:,ig)),psi(:,ig)) 
      enddo

      end subroutine
!======================================================================================



!======================================================================================
      subroutine Init_Psi
!======================================================================================
!     Initialize the wave packet (Generating the Gaussian WP)             
!======================================================================================
      use mainvar
      implicit none
      integer :: i,is
      real*8  :: gaussian

     !setting up grid points in x-space
      do i=1,ngrid
         xgrid(i) = xmin + dx*(i-1)
      enddo

      psi=(0.d0,0.d0)

     !Generating gaussian wavepacket
      sigma_ini=20.d0/k_ini !based on Tully 1990, JCP
      do i=1,ngrid
         gaussian=exp((-(xgrid(i)-xcenter_ini)**2)/(sigma_ini**2))

        !in some case the direction is important, 
        !the sign can be changed depending on the problem in question
         psi(istate_ini,i)=gaussian*exp(jcpx*k_ini*xgrid(i))
      enddo

      call normalize_psi

      end subroutine
!======================================================================================


!======================================================================================
      subroutine Set_Operators
!======================================================================================
!     Building up the Kin and Pot operator part of the Propagator              
!              
!     Since symmetric Trotter splitting is introduced in the 
!     short time propagator, the Op_Pot is only half time step is used
!======================================================================================
      use mainvar
      implicit none
      integer :: i,ig,is,j
      real*8 :: Tt,kx
      real*8, dimension(nstate,nstate) :: Vv
      complex*16, dimension(nstate,nstate) :: Vc,Vc_exp

     !exp[-i*V(x)*dt/2]
      do ig=1,ngrid
      
        !evaluate potential matrix (Vv) at the ith grid point
         call Vpot(xgrid(ig),Vv)

        !the full exponent at dt/2
         Vc=-jcpx*dt*Vv/2.d0

        !compute matrix exponential [exp(Vv)]
         call c8mat_expm1(nstate,Vc,Vc_exp)

         Op_pot(:,:,ig)=Vc_exp

      enddo


     !exp[-iT*dt]
!     Fortran indices run from 1, so the FFT frequency index is i-1. Using i
!     directly shifted the whole momentum grid by one spacing and never produced
!     k = 0, which biased the kinetic propagator.
      do i=1,ngrid
         if(i-1 < ngrid/2) then
           kx=2.d0*pi*dble(i-1)/(dble(ngrid)*dx)
         else
           kx=2.d0*pi*(dble(i-1)-dble(ngrid))/(dble(ngrid)*dx)
         endif

         Tt=0.5d0*kx*kx/redmass

         Op_kin(i)=exp(-jcpx*Tt*dt)
      enddo


      end subroutine
!===========================================================================


!===========================================================================
      subroutine Build_Diabatic_to_Adiabatic_matrix
!===========================================================================
      use mainvar
      implicit none
      integer :: ig 
      real*8, parameter :: trhold=1.d-10
      real*8, dimension(nstate,nstate) :: Vv,eigvec
      real*8, dimension(nstate) :: eigval

      do ig=1,ngrid

        !evaluate potential matrix (Vv) at the ith grid point
         call Vpot(xgrid(ig),Vv)

         call jacobi(Vv,nstate,nstate,eigval,eigvec,trhold,1)
         DtoA(:,:,ig)=eigvec
         Vadiabatic(:,ig)=eigval
      enddo

      end subroutine
!======================================================================================
 

!======================================================================================
      subroutine normalize_psi
!======================================================================================
      use mainvar
      implicit none
      real*8         :: norm
      integer        :: ig,is

      norm=0.0d0

      do is=1,nstate
        !norm of WP in a given state
         do ig=1,ngrid
            norm = norm + dx*psi(is,ig)*conjg(psi(is,ig))
         enddo
      enddo

      psi=psi/sqrt(norm)

      end subroutine
!======================================================================================


!======================================================================================
      subroutine write_psi(istep)
!======================================================================================
      use mainvar
      implicit none
      integer :: n,j ,istep
      integer, save :: call_first=1
      real*8, dimension(nstate) :: PsiPsi
      real*8, dimension(nstate) :: PsiPsi_ad


      if(call_first == 1) then
         write(66,"(3i10)") ngrid,nstep/nwrite,nstate
         write(65,"(3i10)") ngrid,nstep/nwrite,nstate
         call_first=99
      else

         write(66,"(f12.5,a12,f12.5)") istep*dt,"time",k_ini
         write(65,"(f12.5,a12,f12.5)") istep*dt,"time",k_ini
         do n=1,ngrid
            PsiPsi=psi(:,n)*conjg(psi(:,n))
            PsiPsi_ad=psi(:,n)*conjg(psi(:,n))

            write(65,"(f12.5,6e15.5)") xgrid(n),(PsiPsi(j),j=1,nstate)
            write(66,"(f12.5,6e15.5)") xgrid(n),(PsiPsi_ad(j),j=1,nstate)
         enddo

      endif

      flush(66)

      end subroutine
!======================================================================================


!======================================================================================
      subroutine print_checkpoint_psi(istep)
!======================================================================================
      use mainvar
      implicit none
      integer :: n,j ,istep

      write(77,*) "Chekpoint of WP time evolution"
      write(77,"(2i10,3f12.5)") nstate,nstep, dt, redmass,k_ini
      write(77,"(i10,2f12.5)") ngrid,xmin,xmax
      write(77,*)

      write(77,"(f12.5,a12)") istep*dt,"time"
      do n=1,ngrid
         write(77,"(f12.5,10e15.5)") xgrid(n),(Psi(j,ngrid),j=1,nstate)
      enddo

      flush(77)
      rewind(77)

      end subroutine
!======================================================================================



!======================================================================
      subroutine fourier(dir,n,c)
!======================================================================
!     Interface to the double precision FFT in fft_double.f90.
!
!     Previously this called FFTPACK5, whose cfft1f/cfft1b are single
!     precision (complex(kind=4)), while this routine declared its argument
!     complex(kind=8). The mismatch needed -fallow-argument-mismatch and
!     held the whole propagation to single precision.
!
!     dir =  0  no-op, kept so existing call sites still work
!     dir =  1  forward FFT, including the 1/n normalization
!     dir = -1  backward FFT
!     dir =  2  no-op
!======================================================================
      implicit none
      integer(kind=4) :: n,dir
      complex(kind=8) :: c(n)

      if(dir == 1) then
         call dcfft_forward(n,c)
      elseif(dir == -1) then
         call dcfft_backward(n,c)
      endif

      end subroutine fourier
!======================================================================

