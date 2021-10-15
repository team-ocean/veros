
include ../../site_specific.mk_${HOSTTYPE}

all: isoneutral_module.o isoneutral_diffusion.o \
     isoneutral_friction.o isoneutral.o 

clean :
	 rm -f *.o *.mod 

isoneutral_module.o: isoneutral_module.f90 ../main/main_module.o 
	 	$(F90) $(F90FLAGS) -I../main -c isoneutral_module.f90
isoneutral.o: isoneutral.f90 ../main/main_module.o isoneutral_module.o
	 	$(F90) $(F90FLAGS) -I../main -c isoneutral.f90
isoneutral_diffusion.o: isoneutral_diffusion.f90 \
                        ../main/main_module.o isoneutral_module.o
	 	$(F90) $(F90FLAGS) -I../main -c isoneutral_diffusion.f90
isoneutral_friction.o: isoneutral_friction.f90 ../eke/eke_module.o \
                       ../main/main_module.o isoneutral_module.o
	 	$(F90) $(F90FLAGS) -I../main -I../eke -c isoneutral_friction.f90

