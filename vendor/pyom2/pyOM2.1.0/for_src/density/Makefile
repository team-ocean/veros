
include ../../site_specific.mk_${HOSTTYPE}

all:  density.o get_rho.o

check_density : check_density.f90 density.o 
	 $(F90) check_density.f90 density.o $(F90FLAGS) $(CDFFLAGS) \
            -o ../../bin/check_density.x
clean :
	 rm -f *.o *.mod 

density.o: density.f90  
	$(F90) $(F90FLAGS) -c density.f90
get_rho.o: get_rho.f90  ../main/main_module.o  density.o
	$(F90) $(F90FLAGS) -I../main -c get_rho.f90



