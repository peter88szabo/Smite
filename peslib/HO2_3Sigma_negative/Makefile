FC = gfortran
FFLAGS = -O2 -fPIC
TARGET = libho2_3sigma_negative.so
SOURCES = ho2_c_interface.f90 hoopasc_embedded.f

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(FC) -shared $(FFLAGS) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET) *.o *.mod
