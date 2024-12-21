################################################################################
#
#   This Makefile will compile the potential energy surface module to expose it
#   to Python using f2py.
#
################################################################################
export CC=icc
F2PY=f2py

.PHONY: all build pyf clean

all: build

build:
	$(F2PY) -c potnhh.f potnhh.pyf

pyf:
	$(F2PY) -m PES -h potnhh.pyf potnhh.f --overwrite-signature

clean:
	rm -f *.o *.so
