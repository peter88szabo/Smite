FC = gfortran
FFLAGS = -O2 -fPIC -std=legacy -ffixed-line-length-none
TARGET = libho2_1deltag.so
SOURCES = ho2_1deltag_c_interface.f90 ho2expesfast_own_switch_2015_10_19_okk.f

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) ho2pes-excited.dat para-ex.txt
	$(FC) -shared $(FFLAGS) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET) *.o *.mod fort.10
