# Grid-based nonadiabatic wavepacket propagation, used here to produce the
# exact quantum reference for Tully's model I.
#
#   make            build
#   make run        build and run with input.inp in this directory
#   make clean      remove objects, binary and output

FC      := gfortran
FFLAGS  := -O2 -fallow-argument-mismatch
SRC     := src/fft_double.f90 src/jacobi.f src/matrix_exponential.f90 \
           src/WPevolve_nonadiabatic_version-2.03.f90
TARGET  := wavepacket.x

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(FC) $(FFLAGS) $(SRC) -o $@

run: $(TARGET)
	./$(TARGET) < input.inp

clean:
	rm -f $(TARGET) *.o *.mod fort.* *.dat input.out
