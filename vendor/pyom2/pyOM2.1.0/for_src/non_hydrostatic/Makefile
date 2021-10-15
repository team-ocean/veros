
include ../../site_specific.mk_${HOSTTYPE}

all: non_hydrostatic.o 

clean :
	 rm -f *.o *.mod 

non_hydrostatic.o: non_hydrostatic.f90 ../main/main_module.o  
	$(F90) $(F90FLAGS) -I../main -c non_hydrostatic.f90

