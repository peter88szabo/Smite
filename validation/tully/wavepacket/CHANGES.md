# Changes to the wavepacket code

Imported from `WP_dynamics_for_nonadiabatic/.../NonAdiabatic-Wavepacket/src`
(version 2.03) and corrected. The original is kept beside it as
`src/WPevolve_nonadiabatic_version-2.03.f90.orig`.

## 1. Momentum grid was shifted by one index

This is the only change that altered a result.

```fortran
do i=1,ngrid
   if(i < ngrid/2) then
      kx = 2*pi*float(i)/(float(ngrid)*dx)      ! <- i, not i-1
```

Fortran indexes from 1, so the FFT frequency index is `i-1`. Using `i` shifted
the whole momentum grid by one spacing `2*pi/(N*dx)` and never produced `k = 0`:

| index | 1 | 2 | 3 | 4 | ... | N |
| --- | --- | --- | --- | --- | --- | --- |
| code | 0.785 | 1.571 | 2.356 | -3.142 | ... | 0.000 |
| correct | 0.000 | 0.785 | 1.571 | 2.356 | ... | -0.785 |

Every momentum in the kinetic propagator `exp(-i k^2 dt / 2m)` was therefore
displaced, biasing both the group velocity and the accumulated phase.

**Effect on Tully model I**, 2048 points on [-40,40], 3000 steps of dt=1,
starting at x=-6 on the lower surface with reduced mass 2000:

| k | P(upper), before | P(upper), after |
| --- | --- | --- |
| 10 | 0.15774 | 0.15523 |
| 15 | 0.32586 | 0.32304 |

About 1.6% relative, in the same direction at both energies.

## 2. Everything is now double precision

`complex*8` is **single** precision in Fortran, not double: kind 4, eps 1.2e-7.
Eight declarations used it -- the wavefunction, both propagator operators, the
diabatic-to-adiabatic matrix -- while every real in the program was `real*8`.
`Set_Operators` even built the potential propagator in `complex*16` and then
truncated it on assignment.

The FFT was the reason it could not simply be promoted: FFTPACK5's `cfft1f` and
`cfft1b` are single precision (`complex(kind=4)`), while `fourier` declared its
argument `complex(kind=8)`. That mismatch is what made
`-fallow-argument-mismatch` necessary. It was harmless in practice, because
`fourier` performed no arithmetic on the array and simply passed the address to
a single-precision routine that read exactly the caller's single-precision
bytes, but it capped the whole propagation at single precision.

FFTPACK5 has been replaced by `src/fft_double.f90`, a self-contained double
precision radix-2 transform checked against NumPy: agreement to the last digit
on the forward transform and a round-trip error of 9e-16. The 2048-point grid
is a power of two, which is what radix 2 requires.

**This changed nothing numerically.** Double precision with the original
momentum grid reproduces the original results to five digits, so single
precision was never costing accuracy here; the split-operator scheme is
well-conditioned and the per-step renormalization absorbed the roundoff. It is
fixed for robustness, not because it was distorting anything.

`c8mat_expm1` and `jacobi` were already double and are untouched.

## Not changed

`normalize_psi` is still called every step. Split-operator propagation is
unitary, so a drifting norm would be a useful warning -- of roundoff, or of the
packet reaching the periodic boundary and wrapping. Renormalizing hides it. The
grids used here are wide enough that nothing wraps, so this is a remark rather
than a fault.

## Building and running

```bash
make
make run          # reads input.inp on stdin
```
