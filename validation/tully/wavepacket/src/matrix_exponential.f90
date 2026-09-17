!==============================================================
subroutine c8mat_expm1(n,a,e)
!==============================================================
!  C8MAT_EXPM1 is essentially MATLAB's built-in matrix exponential algorithm.
!  Licensing:
!    This code is distributed under the GNU LGPL license.
!  Modified:
!    01 March 2013
!  Author:
!    FORTRAN90 version by John Burkardt
!  Reference:
!    Cleve Moler, Charles VanLoan,
!    Nineteen Dubious Ways to Compute the Exponential of a Matrix,
!    Twenty-Five Years Later,
!    SIAM Review,
!    Volume 45, Number 1, March 2003, pages 3-49.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the dimension of the matrix.
!
!    Input, complex ( kind = 8 ) A(N,N), the matrix.
!
!    Output, complex ( kind = 8 ) E(N,N), the estimate for exp(A).
!
  implicit none

  integer ( kind = 4 ) n
  complex ( kind = 8 ) a(n,n), a2(n,n)
  real ( kind = 8 ) a_norm, c, c8mat_norm_li
  complex ( kind = 8 ) d(n,n), e(n,n)
  integer ( kind = 4 ) ee,k
  logical p
  integer ( kind = 4 ) , parameter :: q = 6
  real ( kind = 8 ) r8_log_2
  integer ( kind = 4 ) s
  complex ( kind = 8 ) x(n,n)
 
!  Make a copy of the matrix.
  a2(1:n,1:n) = a(1:n,1:n)
 
!  Compute the L-infinity norm.
  a_norm = c8mat_norm_li ( n, n, a2 )
 
!  Determine a scaling factor for the matrix.
  ee = int ( r8_log_2 ( a_norm ) ) + 1

  s = max ( 0, ee + 1 )

  a2(1:n,1:n) = a2(1:n,1:n) / 2.0D+00 ** s

  x(1:n,1:n) = a2(1:n,1:n)

  c = 0.5D+00

  call c8mat_identity ( n, e )
  e(1:n,1:n) = e(1:n,1:n) + c * a2(1:n,1:n)

  call c8mat_identity ( n, d )
  d(1:n,1:n) = d(1:n,1:n) - c * a2(1:n,1:n)

  p = .true.

  do k = 2, q

    c = c * real ( q - k + 1, kind = 8 ) &
      / real ( k * ( 2 * q - k + 1 ), kind = 8 )

    x(1:n,1:n) = matmul ( a2(1:n,1:n), x(1:n,1:n) )

    e(1:n,1:n) = e(1:n,1:n) + c * x(1:n,1:n)

    if ( p ) then
      d(1:n,1:n) = d(1:n,1:n) + c * x(1:n,1:n)
    else
      d(1:n,1:n) = d(1:n,1:n) - c * x(1:n,1:n)
    end if

    p = .not. p

  end do
!
!  E -> inverse(D) * E
!
  call c8mat_minvm ( n, n, d, e, e )
!
!  E -> E^(2*S)
!
  do k = 1, s
    e(1:n,1:n) = matmul ( e(1:n,1:n), e(1:n,1:n) )
  end do

  return
end
!==============================================================



subroutine c8mat_fss ( n, a, nb, b, info )

!*****************************************************************************80
!
!! C8MAT_FSS factors and solves a system with multiple right hand sides.
!
!  Discussion:
!
!    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].
!
!    This routine does not save the LU factors of the matrix, and hence cannot
!    be used to efficiently solve multiple linear systems, or even to
!    factor A at one time, and solve a single linear system at a later time.
!
!    This routine uses partial pivoting, but no pivot vector is required.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    01 March 2013
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the order of the matrix.
!    N must be positive.
!
!    Input/output, complex ( kind = 8 ) A(N,N).
!    On input, A is the coefficient matrix of the linear system.
!    On output, A is in unit upper triangular form, and
!    represents the U factor of an LU factorization of the
!    original coefficient matrix.
!
!    Input, integer ( kind = 4 ) NB, the number of right hand sides.
!
!    Input/output, complex ( kind = 8 ) B(N,NB).
!    On input, the right hand sides of the linear system.
!    On output, the solutions of the linear systems.
!
!    Output, integer ( kind = 4 ) INFO, singularity flag.
!    0, no singularity detected.
!    nonzero, the factorization failed on the INFO-th step.
!
  implicit none

  integer ( kind = 4 ) n
  integer ( kind = 4 ) nb

  complex ( kind = 8 ) a(n,n)
  complex ( kind = 8 ) b(n,nb)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) info
  integer ( kind = 4 ) ipiv
  integer ( kind = 4 ) j
  integer ( kind = 4 ) jcol
  real ( kind = 8 ) piv
  complex ( kind = 8 ) row(n)
  complex ( kind = 8 ) t(nb)
  complex ( kind = 8 ) temp

  info = 0

  do jcol = 1, n
!
!  Find the maximum element in column I.
!
    piv = abs ( a(jcol,jcol) )
    ipiv = jcol
    do i = jcol + 1, n
      if ( piv < abs ( a(i,jcol) ) ) then
        piv = abs ( a(i,jcol) )
        ipiv = i
      end if
    end do

    if ( piv == 0.0D+00 ) then
      info = jcol
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'C8MAT_FSS - Fatal error!'
      write ( *, '(a,i8)' ) '  Zero pivot on step ', info
      stop 1
    end if
!
!  Switch rows JCOL and IPIV, and B.
!
    if ( jcol /= ipiv ) then

      row(1:n) = a(jcol,1:n)
      a(jcol,1:n) = a(ipiv,1:n)
      a(ipiv,1:n) = row(1:n)

      t(1:nb)      = b(jcol,1:nb)
      b(jcol,1:nb) = b(ipiv,1:nb)
      b(ipiv,1:nb) = t(1:nb)

    end if
!
!  Scale the pivot row.
!
    a(jcol,jcol+1:n) = a(jcol,jcol+1:n) / a(jcol,jcol)
    b(jcol,1:nb) = b(jcol,1:nb) / a(jcol,jcol)
    a(jcol,jcol) = 1.0D+00
!
!  Use the pivot row to eliminate lower entries in that column.
!
    do i = jcol + 1, n
      if ( a(i,jcol) /= 0.0D+00 ) then
        temp = - a(i,jcol)
        a(i,jcol) = 0.0D+00
        a(i,jcol+1:n) = a(i,jcol+1:n) + temp * a(jcol,jcol+1:n)
        b(i,1:nb) = b(i,1:nb) + temp * b(jcol,1:nb)
      end if
    end do

  end do
!
!  Back solve.
!
  do j = 1, nb
    do jcol = n, 2, -1
      b(1:jcol-1,j) = b(1:jcol-1,j) - a(1:jcol-1,jcol) * b(jcol,j)
    end do
  end do

  return
end
subroutine c8mat_identity ( n, a )

!*****************************************************************************80
!
!! C8MAT_IDENTITY sets a C8MAT to the identity.
!
!  Discussion:
!
!    A C8MAT is a matrix of C8's.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    08 June 2008
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the order of the matrix.
!
!    Output, complex ( kind = 8 ) A(N,N), the matrix.
!
  implicit none

  integer ( kind = 4 ) n

  complex ( kind = 8 ) a(n,n)
  integer ( kind = 4 ) i

  a(1:n,1:n) = cmplx ( 0.0D+00, 0.0D+00, kind = 8 )

  do i = 1, n
    a(i,i) = cmplx ( 1.0D+00, 0.0D+00, kind = 8 )
  end do

  return
end

subroutine c8mat_minvm ( n1, n2, a, b, c )
!*****************************************************************************80
!  C8MAT_MINVM computes inverse(A) * B for C8MAT's.
!  Discussion:
!    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].
!  Licensing:
!    This code is distributed under the GNU LGPL license.
!  Modified:
!    01 March 2013
!  Author:
!    John Burkardt
!  Parameters:
!
!    Input, integer ( kind = 4 ) N1, N2, the order of the matrices.
!
!    Input, complex ( kind = 8 ) A(N1,N1), B(N1,N2), the matrices.
!
!    Output, complex ( kind = 8 ) C(N1,N2), the result, C = inverse(A) * B.
!
  implicit none

  integer ( kind = 4 ) n1
  integer ( kind = 4 ) n2

  complex ( kind = 8 ) a(n1,n1)
  complex ( kind = 8 ) alu(n1,n1)
  complex ( kind = 8 ) b(n1,n2)
  complex ( kind = 8 ) c(n1,n2)
  integer ( kind = 4 ) info

  alu(1:n1,1:n1) = a(1:n1,1:n1)
  c(1:n1,1:n2) = b(1:n1,1:n2)

  call c8mat_fss ( n1, alu, n2, c, info )
 
  if ( info /= 0 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'C8MAT_MINVM - Fatal error!'
    write ( *, '(a)' ) '  The matrix A was numerically singular.'
    stop 1
  end if

  return
end

function c8mat_norm_li ( m, n, a )

!*****************************************************************************80
!
!! C8MAT_NORM_LI returns the matrix L-oo norm of a C8MAT.
!
!  Discussion:
!
!    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].
!
!    The matrix L-oo norm is defined as:
!
!      C8MAT_NORM_LI =  max ( 1 <= I <= M ) sum ( 1 <= J <= N ) abs ( A(I,J) ).
!
!    The matrix L-oo norm is derived from the vector L-oo norm,
!    and satisifies:
!
!      c8vec_norm_li ( A * x ) <= c8mat_norm_li ( A ) * c8vec_norm_li ( x ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    01 March 2013
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) M, the number of rows in A.
!
!    Input, integer ( kind = 4 ) N, the number of columns in A.
!
!    Input, complex ( kind = 8 ) A(M,N), the matrix whose L-oo
!    norm is desired.
!
!    Output, real ( kind = 8 ) C8MAT_NORM_LI, the L-oo norm of A.
!
  implicit none

  integer ( kind = 4 ) m
  integer ( kind = 4 ) n

  complex ( kind = 8 ) a(m,n)
  real ( kind = 8 ) c8mat_norm_li
  integer ( kind = 4 ) i
  real ( kind = 8 ) row_sum

  c8mat_norm_li = 0.0D+00

  do i = 1, m
    row_sum = sum ( abs ( a(i,1:n) ) )
    c8mat_norm_li = max ( c8mat_norm_li, row_sum )
  end do

  return
end


!==============================================================
function r8_log_2 ( x )
!==============================================================
!! R8_LOG_2 returns the logarithm base 2 of an R8.
!  Discussion:
!    value = Log ( |X| ) / Log ( 2.0 )
!  Licensing:
!    This code is distributed under the GNU LGPL license.
!  Modified:
!    27 August 2002
!  Author:
!    John Burkardt
!  Parameters:
!    Input, real ( kind = 8 ) X, the number whose base 2 logarithm is desired.
!    X should not be 0.
!
!    Output, real ( kind = 8 ) R8_LOG_2, the logarithm base 2 of the absolute
!    value of X.  It should be true that |X| = 2^R8_LOG_2.
!
  implicit none
  real ( kind = 8 ) r8_log_2
  real ( kind = 8 ) x

  if ( x == 0.0D+00 ) then
    r8_log_2 = - huge ( x )
  else
    r8_log_2 = log ( abs ( x ) ) / log ( 2.0D+00 )
  end if

  return
end
!==============================================================

