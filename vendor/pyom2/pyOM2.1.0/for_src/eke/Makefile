
include ../../site_specific.mk_${HOSTTYPE}

all: eke_module.o eke.o  

eke_module.o: eke_module.f90 ../main/main_module.o 
	 	$(F90) $(F90FLAGS) -I../main -c eke_module.f90
eke.o: eke.f90 ../main/main_module.o \
               ../isoneutral/isoneutral_module.o eke_module.o
	 	$(F90) $(F90FLAGS) -I../main -I../isoneutral -c eke.f90
clean :
	 rm -f *.o *.mod 

