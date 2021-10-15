
include ../../site_specific.mk_${HOSTTYPE}

all: main_module.o friction.o numerics.o advection.o diffusion.o\
     momentum.o thermodynamics.o restart.o main.o

clean :
	 rm -f *.o *.mod 

main_module.o : main_module.f90 
	$(F90) $(F90FLAGS) -c main_module.f90
numerics.o: numerics.f90 main_module.o
	$(F90) $(F90FLAGS) -c numerics.f90
friction.o: friction.f90 main_module.o 
	$(F90) $(F90FLAGS) -c friction.f90
advection.o: advection.f90 main_module.o 
	$(F90) $(F90FLAGS) -c advection.f90
diffusion.o: diffusion.f90 main_module.o 
	$(F90) $(F90FLAGS) -c diffusion.f90

main.o: main.f90 main_module.o ../tke/tke_module.o ../eke/eke_module.o \
        ../idemix/idemix_module.o ../diagnostics/diagnostics_module.o \
        ../etc/timing_module.o
	$(F90) $(F90FLAGS) -I../tke -I../eke -I../idemix -I../diagnostics -I../etc -c main.f90

momentum.o: momentum.f90 main_module.o ../isoneutral/isoneutral_module.o ../etc/timing_module.o
	$(F90) $(F90FLAGS) -I../isoneutral -I../etc -c momentum.f90

thermodynamics.o: thermodynamics.f90 main_module.o \
                  ../isoneutral/isoneutral_module.o ../tke/tke_module.o \
                  ../density/density.o ../etc/timing_module.o
	$(F90) $(F90FLAGS) -I../isoneutral -I../etc -I../density -I../tke -c thermodynamics.f90

restart.o: restart.f90  main_module.o ../tke/tke_module.o ../eke/eke_module.o \
           ../idemix/idemix_module.o
	$(F90) $(F90FLAGS) -I../tke -I../eke -I../idemix -c restart.f90

