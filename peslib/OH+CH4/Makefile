################################################################################
#
#   This Makefile will compile the potential energy surface module to expose it
#   to Python using f2py.
#
################################################################################

F2PY=f2py
FC=gfortran
FFLAGS=-O2 -fPIC
SHARED_TARGET=libohch4_pes.so

.PHONY: all shared build pyf clean

all: shared

shared:
	$(FC) -shared $(FFLAGS) -o $(SHARED_TARGET) ohch4_c_interface.f90 ch4oh.f utility.f

build:
	$(F2PY) -c ch4oh.f utility.f ch4oh.pyf

pyf:
	$(F2PY) -m PES -h ch4oh.pyf ch4oh.f --overwrite-signature

clean:
	rm -f *.o *.mod *.so $(SHARED_TARGET)
