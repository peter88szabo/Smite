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
	$(F2PY) -c oh2jw.f utility.f oh2jw.pyf

pyf:
	$(F2PY) -m PES -h oh2jw.pyf oh2jw.f --overwrite-signature

clean:
	rm -f *.o PES.so
