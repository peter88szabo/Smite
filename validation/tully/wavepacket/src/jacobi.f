!===========================================================================
      subroutine jacobi(a,nmax,n,d,x,th,iord)                          
!===========================================================================
!  threshold jacobian method for eigenvalues of symmetric matricies        !
!===========================================================================
! a(*,*)--> input matrix to be diagonalized                                !
! nmax  --> maximum order of the problem                                   !
! n     --> actual order                                                   !
! th    --> treshold                                                       !
! iord  --> sorting variable, +1: increasing order, -1: decrecasing order  !
! x(*,*)--> matrix of the eigenvectors                                     !
! d(*)  --> block of the eigenvalues                                       !
! itmax --> maximum number of the jacobian rotations                       ! 
!===========================================================================

      implicit real*8 (a-h,o-z)
      parameter (itmax=50)
      dimension a(nmax,nmax),x(nmax,nmax),d(nmax)

      do i=1,n
        do j=1,n
         x(i,j)=0.0d0
        enddo
         x(i,i)=1.d0
      enddo

      do i=1,n
        d(i)=a(i,i)
      enddo
!----------------------------------------------------
      do iter=1,itmax
       amax=0.0d0
        do i=2,n
         do 10 j=1,i-1
           aii=d(i)
           ajj=d(j)
           aij=a(i,j)
           if (dabs(aij) .gt. amax) amax=dabs(aij)
           if (dabs(aij) .le. th) goto 10 
           alpha=0.5d0*(aii-ajj)/aij
!   t=s/c     
           t=-alpha+dsqrt(1.d0+alpha*alpha)
           if (alpha .lt. 0.d0)  t=-alpha-dsqrt(1.d0+alpha*alpha)
           c=1.0d0/dsqrt(1.d0 + t*t)
           s=c*t
            
           do 20 k=1,n
              xj=c*x(k,j)-s*x(k,i)
              x(k,i)=s*x(k,j)+c*x(k,i)
              x(k,j)=xj
c             x(k,j)=s*x(k,j)+c*x(k,i)
c             x(k,i)=xj
              if (k .eq. j) goto 20
              if (k .lt. j) then
                xj=c*a(j,k)-s*a(i,k)
                a(i,k)=s*a(j,k)+c*a(i,k)
                a(j,k)=xj
                goto 20
              endif

              if (k .eq. i) goto 20
              if (k .lt. i) then
                xj=c*a(k,j)-s*a(i,k)
                a(i,k)=s*a(k,j)+c*a(i,k)
                a(k,j)=xj
                goto 20
              endif
              xj=c*a(k,j)-s*a(k,i)
              a(k,i)=s*a(k,j)+c*a(k,i)
              a(k,j)=xj
  20       continue
           d(i)=c*c*aii+s*s*ajj+2.d0*s*c*aij
           d(j)=c*c*ajj+s*s*aii-2.d0*s*c*aij
           a(i,j)=0.0d0
  10     continue
       enddo !end of i loop

      if (amax .le. th) goto 30

      enddo !end of iter loop
  30  continue
!--------------------------------------------------      
      if (iord .eq. 1) then
!     arrange eigenvalues in increasing order      
      
      do k=1,n-1
        dmn=d(k)
        kmin=k
        do j=k+1,n
          if (dmn .gt. d(j) ) then 
            kmin=j
            dmn=d(j)
          endif
        enddo
        if (k .ne. kmin) then
          do j=1,n
           call swap( x(j,kmin),x(j,k) )
          enddo
           call swap( d(kmin),d(k) )
        endif
      enddo
!------------------------------------------------
      elseif (iord .eq. -1) then
!     arrange egeinvalues in decreasing order      
      do k=1,n-1
        dmx=d(k)
        kmax=k
        do j=k+1,n
          if (dmx .lt. d(j)) then
            kmax=j
            dmx=d(j)
          endif
        enddo
         if (k .ne. kmax) then
           do j=1,n
            call swap(x(j,kmax), x(j,k))
           enddo
            call swap(d(kmax),d(k))
         endif
      enddo
      endif
!----------------------------------------------
!   restore the original a(i,j) matrix
      do i=1,n
       do j=1,i-1
         a(i,j)=a(j,i)
       enddo  
      enddo
!----------------------------------------------   
      return
      end subroutine
!====================================================================
      subroutine swap(a,b)
      implicit real*8 (a-h,o-z)
       
      temp=a
      a=b
      b=temp
      return
      end subroutine
!====================================================================



