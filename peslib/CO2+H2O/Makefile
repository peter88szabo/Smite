#=====================User define area===================
MODULE_NAME = pes        # Name of the Python module
FORTRAN_SRCS = \
pes_co2h2o_sr.f90 \
pes_co2h2o_lr_basis.f90 \
pes_co2h2o_lr.f90 \
pes_co2h2o_2b.f90
OUTPUT = $(MODULE_NAME).so
LIB = -Llib/pes-xyz -lpes
MOD = -Ilib/mod
#========================================================

F2PY = f2py
FC = ifx

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(FORTRAN_SRCS)
	$(F2PY) -c -m $(MODULE_NAME) $(FORTRAN_SRCS) $(LIB) $(MOD) --f90flags="-fPIC -r8 $(MOD) $(LIB)"

clean:
	rm -f *.so *.o *.mod

