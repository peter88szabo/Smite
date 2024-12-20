ifx -fPIC -shared -r8 -o libpesco2h2o.so pes_co2h2o_sr.f90 pes_co2h2o_lr_basis.f90 pes_co2h2o_lr.f90 pes_co2h2o_2b.f90 -I lib/mod -L lib/pes-xyz 

rm -rf *.o *.mod

