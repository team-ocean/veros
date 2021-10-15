
target = with_mpi
code   = ../../for_src
bin    = ../../bin

%.o: %.f90
	cp $< ${code}/config.f90
	(cd ${code};make $(target) )

%: %.o
	mv ${bin}/model.x ${@}.x

clean: 
	(cd ${code}; make clean)


