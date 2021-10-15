
include ../../site_specific.mk_${HOSTTYPE}

all: solve_pressure.o streamfct_init.o solve_streamfct.o island_module.o  

clean :
	 rm -f *.o *.mod 

island_module.o: island_module.f90  
	$(F90) $(F90FLAGS) -c island_module.f90
solve_pressure.o: solve_pressure.f90 ../main/main_module.o 
	$(F90) $(F90FLAGS) -I../main -c solve_pressure.f90
streamfct_init.o: streamfct_init.f90 ../main/main_module.o  
	$(F90) $(F90FLAGS) -I../main -c streamfct_init.f90
solve_streamfct.o: solve_streamfct.f90 ../main/main_module.o  
	$(F90) $(F90FLAGS) -I../main -c solve_streamfct.f90

