
include ../../site_specific.mk_${HOSTTYPE}

all: timing_module.o py_stuff.o 

check_timing : check_timing.f90 timing_module.o
	 $(F90) check_timing.f90 timing_module.o $(F90FLAGS) \
           -o ../../bin/check_timing.x

clean :
	 rm -f *.o *.mod 


timing_module.o: timing_module.f90  
	$(F90) $(F90FLAGS) -c timing_module.f90
py_stuff.o: py_stuff.f90
	$(F90) $(F90FLAGS) -c py_stuff.f90

