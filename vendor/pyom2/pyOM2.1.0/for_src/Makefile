
include ../site_specific.mk_${HOSTTYPE}

default: with_mpi

modules:
	cd etc; make timing_module.o
	cd density; make density.o
	cd main; make main_module.o
	cd eke; make eke_module.o
	cd isoneutral; make isoneutral_module.o
	cd idemix; make idemix_module.o
	cd tke; make tke_module.o
	cd diagnostics; make diagnostics_module.o

all:    modules
	cd main; make 
	cd tke; make
	cd eke; make
	cd idemix; make
	cd isoneutral; make
	cd external; make
	cd non_hydrostatic; make
	cd density; make
	cd diagnostics; make 

config.o: all config.f90 
	$(F90) $(F90FLAGS) $(CDFFLAGS) -Imain -Iisoneutral -Itke -Ieke \
	 -Iidemix -Idiagnostics -Idensity -c config.f90

with_mpi: all config.o
	cd parallel; make parallel_mpi.o
	$(F90) main/*.o tke/*.o eke/*.o idemix/*.o isoneutral/*.o \
	 external/*.o non_hydrostatic/*.o density/*.o diagnostics/*.o \
	 config.o parallel/parallel_mpi.o etc/timing_module.o\
	 $(F90FLAGS) $(MPIFLAGS) $(CDFFLAGS)-o ../bin/model.x

without_mpi: all config.o
	cd parallel; make parallel_none.o
	$(F90) main/*.o tke/*.o eke/*.o idemix/*.o isoneutral/*.o \
	 external/*.o non_hydrostatic/*.o density/*.o diagnostics/*.o \
	 config.o parallel/parallel_none.o etc/timing_module.o\
	 $(F90FLAGS) $(CDFFLAGS)-o ../bin/model.x

dirs = density diagnostics eke etc external idemix isoneutral main \
       non_hydrostatic parallel tke
clean: 
	for d in $(dirs); do cd $$d && make clean && cd ..; done
	rm -f *.o *.mod 


