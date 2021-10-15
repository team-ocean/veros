
include ../../site_specific.mk_${HOSTTYPE}

parallel_none.o: parallel_none.f90 ../main/main_module.o 
	$(F90) $(F90FLAGS) -I../main -c parallel_none.f90
parallel_mpi.o: parallel_mpi.f90 ../main/main_module.o  
	$(F90) $(F90FLAGS) $(MPIFLAGS) -I../main -c parallel_mpi.f90
clean :
	 rm -f *.o *.mod 

