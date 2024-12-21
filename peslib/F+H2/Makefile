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
	$(F2PY) -c fh2n5z.f utility.f fh2n5z.pyf

pyf:
	$(F2PY) -m PES -h fh2n5z.pyf fh2n5z.f --overwrite-signature

clean:
	rm -f *.o PES.so
