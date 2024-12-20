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
	$(F2PY) -c ch4cl2005.f utility.f ch4cl2005_new.pyf 

pyf:
	$(F2PY) -m PES -h ch4cl2005_new.pyf ch4cl2005.f --overwrite-signature

clean:
	rm -f *.o PES.so
