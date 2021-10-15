




program check_timing
 use timing_module
 implicit none
 integer :: i,j
 real :: fxa
 real,external :: timing_secs2

 call tic('loop')
 do j=1,100000
  do i=1,100000
    fxa = sin(2.7*3.1415)*cos(2.88*3.1415)+1.2
  enddo
 enddo
 print*,' fxa  = ',fxa
 call toc('loop')
 print*,' ok done'

  call get_timing('loop',fxa)
  print*,' costs  = ',fxa,' s'
  call get_timing('tictoc',fxa)
  print*,' tictoc  = ',fxa,' s'
  call get_timing('test',fxa)
  print*,' test  = ',fxa,' s'

  print*,' costs  = ',timing_secs('loop'),' s'
  print*,' tictoc = ',timing_secs('tictoc'),' s'
  print*,' test = ',timing_secs('test'),' s'

end program


