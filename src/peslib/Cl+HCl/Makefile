################################################################################
#
#   This Makefile will compile the potential energy surface module to expose it
#   to Python using f2py.
#
################################################################################

F2PY=f2py

.PHONY: all build pyf clean

all: build

build:
	$(F2PY) -c clhclbcmr.f utility.f clhclbcmr.pyf

pyf:
	$(F2PY) -m PES -h clhclbcmr.pyf clhclbcmr.f --overwrite-signature

clean:
	rm -f *.o PES.so
