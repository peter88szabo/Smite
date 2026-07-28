################################################################################
#
#   This Makefile will compile the potential energy surface module to expose it
#   to Python using f2py.
#
################################################################################

F2PY=f2py
FC=gfortran
FFLAGS=-O2 -fPIC
SHARED_TARGET=libclch4_pes.so

.PHONY: all shared build pyf clean

all: shared

shared:
	$(FC) -shared $(FFLAGS) -o $(SHARED_TARGET) clch4_c_interface.f90 ch4cl2005.f ../F+H2/utility.f

build:
	$(F2PY) -c ch4cl2005.f utility.f ch4cl2005_new.pyf 

pyf:
	$(F2PY) -m PES -h ch4cl2005_new.pyf ch4cl2005.f --overwrite-signature

clean:
	rm -f *.o *.mod PES*.so $(SHARED_TARGET)
