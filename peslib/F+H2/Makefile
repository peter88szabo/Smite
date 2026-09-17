################################################################################
#
#   This Makefile will compile the potential energy surface module to expose it
#   to Python using f2py.
#
################################################################################

F2PY=f2py
FC=gfortran
FFLAGS=-O2 -fPIC
SHARED_TARGET=libfh2_pes.so

.PHONY: all shared build pyf clean

all: shared

shared:
	$(FC) -shared $(FFLAGS) -o $(SHARED_TARGET) fh2_c_interface.f90 fh2n5z.f utility.f

build:
	$(F2PY) -c fh2n5z.f utility.f fh2n5z.pyf

pyf:
	$(F2PY) -m PES -h fh2n5z.pyf fh2n5z.f --overwrite-signature

clean:
	rm -f *.o *.mod PES*.so $(SHARED_TARGET)
