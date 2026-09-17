!======================================================================
!     Double precision complex FFT, radix 2.
!
!     Replaces the FFTPACK5 routines the code used previously. Those are
!     single precision -- cfft1f declares its argument complex(kind=4) --
!     which capped the whole propagation at about seven digits while every
!     other quantity in the program was double.
!
!     Conventions match what the caller expected of FFTPACK5:
!
!       forward   c(k) = (1/n) sum_j c(j) exp(-2 pi i j k / n)
!       backward  c(j) =       sum_k c(k) exp(+2 pi i j k / n)
!
!     so a forward followed by a backward returns the input unchanged. The
!     length must be a power of two, which the 2048-point grid used here is.
!======================================================================

      subroutine dcfft(n, c, sign_exponent, scale)
!======================================================================
!     In-place iterative Cooley-Tukey transform.
!       sign_exponent = -1 forward, +1 backward
!       scale         = factor applied to the result
!======================================================================
      implicit none
      integer, intent(in) :: n, sign_exponent
      real(kind=8), intent(in) :: scale
      complex(kind=8), intent(inout) :: c(n)

      real(kind=8), parameter :: pi = 3.141592653589793238462643d0
      integer :: i, j, k, m, step, bits
      complex(kind=8) :: temp, w, wm
      real(kind=8) :: angle

!     The length has to be a power of two for radix 2.
      bits = 0
      m = n
      do while (m > 1)
         if (mod(m, 2) /= 0) then
            write(*,*) 'dcfft: length must be a power of two, got ', n
            stop
         endif
         m = m / 2
         bits = bits + 1
      enddo

!     Bit-reversal permutation.
      j = 0
      do i = 0, n - 2
         if (i < j) then
            temp = c(i + 1)
            c(i + 1) = c(j + 1)
            c(j + 1) = temp
         endif
         m = n / 2
         do while (m >= 1 .and. j >= m)
            j = j - m
            m = m / 2
         enddo
         j = j + m
      enddo

!     Butterflies, doubling the block length each pass.
      step = 1
      do while (step < n)
         angle = sign_exponent * pi / dble(step)
         wm = dcmplx(dcos(angle), dsin(angle))
         do m = 0, step - 1
            w = dcmplx(dcos(m * angle), dsin(m * angle))
            do i = m, n - 1, 2 * step
               k = i + step
               temp = w * c(k + 1)
               c(k + 1) = c(i + 1) - temp
               c(i + 1) = c(i + 1) + temp
            enddo
         enddo
         step = 2 * step
      enddo

      if (scale /= 1.0d0) c = c * scale

      end subroutine dcfft
!======================================================================


!======================================================================
      subroutine dcfft_forward(n, c)
!======================================================================
      implicit none
      integer, intent(in) :: n
      complex(kind=8), intent(inout) :: c(n)

      call dcfft(n, c, -1, 1.0d0 / dble(n))

      end subroutine dcfft_forward
!======================================================================


!======================================================================
      subroutine dcfft_backward(n, c)
!======================================================================
      implicit none
      integer, intent(in) :: n
      complex(kind=8), intent(inout) :: c(n)

      call dcfft(n, c, 1, 1.0d0)

      end subroutine dcfft_backward
!======================================================================
