
include ../../site_specific.mk_${HOSTTYPE}

all: tke_module.o tke.o

tke_module.o: tke_module.f90 ../main/main_module.o 
	 	$(F90) $(F90FLAGS) -I../main -c tke_module.f90
tke.o: tke.f90 ../main/main_module.o tke_module.o \
               ../eke/eke_module.o ../idemix/idemix_module.o
	 	$(F90) $(F90FLAGS) -I../main -I../idemix -I../eke -c tke.f90
clean :
	 rm -f *.o *.mod 

