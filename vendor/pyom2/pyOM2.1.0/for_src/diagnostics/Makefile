
include ../../site_specific.mk_${HOSTTYPE}

all: diagnostics_module.o diag_main.o diag_averages.o diag_over.o diag_snap.o \
     diag_energy.o diag_particles.o diag_particles_netcdf.o

clean :
	 rm -f *.o *.mod 

diagnostics_module.o: diagnostics_module.f90
	$(F90) $(F90FLAGS) -c diagnostics_module.f90

diag_main.o: diagnostics_module.f90 ../main/main_module.o ../tke/tke_module.o \
             ../isoneutral/isoneutral_module.o ../eke/eke_module.o \
             ../idemix/idemix_module.o
	$(F90) $(F90FLAGS) $(CDFFLAGS) -I../main -I../tke -I../isoneutral -I../eke -I../idemix -c diag_main.f90

diag_averages.o: diag_averages.f90 ../main/main_module.o diagnostics_module.o
	$(F90) $(F90FLAGS) $(CDFFLAGS) -I../main -c diag_averages.f90

diag_energy.o: diag_energy.f90 ../main/main_module.o ../tke/tke_module.o ../isoneutral/isoneutral_module.o \
               ../eke/eke_module.o ../idemix/idemix_module.o diagnostics_module.o
	$(F90) $(F90FLAGS) $(CDFFLAGS) -I../main -I../tke -I../eke -I../idemix -I../isoneutral -c diag_energy.f90

diag_snap.o: diag_snap.f90 ../main/main_module.o ../tke/tke_module.o \
             ../isoneutral/isoneutral_module.o ../eke/eke_module.o \
             ../idemix/idemix_module.o diagnostics_module.o
	$(F90) $(F90FLAGS) $(CDFFLAGS) -I../main -I../tke -I../eke -I../idemix -I../isoneutral -c diag_snap.f90

diag_over.o: diag_over.f90 ../main/main_module.o ../isoneutral/isoneutral_module.o 
	$(F90) $(F90FLAGS) $(CDFFLAGS) -I../main -I../isoneutral -c diag_over.f90

diag_particles.o: diag_particles.f90 ../main/main_module.o 
	$(F90) $(F90FLAGS) -I../main -c diag_particles.f90
diag_particles_netcdf.o: diag_particles_netcdf.f90 ../main/main_module.o  diag_particles.o
	$(F90) $(F90FLAGS) $(CDFFLAGS) -I../main -c diag_particles_netcdf.f90
