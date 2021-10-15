

!=======================================================================
! timing module
! stores elapsed time between calls of tic/toc subroutines
! there is the need to specify a time measurement routine below
!=======================================================================

 module timing_module
 implicit none
 integer, parameter, private :: max_counter = 1000
 integer,private             :: act_counter = 0
 character (len=80),private  :: ref_counter(max_counter)
 real,private                :: counter(0:max_counter)=0,now(0:max_counter)=0

 contains

 real function elapsed()
 implicit none
!
!  elapsed should give the elasped cpu (user) time
!  a function which gives elapsed cpu time is needed here
!
! this works for linux and many other unix machines
       !real (kind=4) :: tarray(2),fxa
       !call etime(tarray,fxa)
       !elapsed = tarray(1)
       real (kind=4) :: fxa
       call cpu_time(fxa)
       elapsed = fxa
 end function elapsed


 subroutine tic(ref)
 implicit none
 character (len=*), intent(in)  :: ref
 integer :: n,len
 now(0)=elapsed()
 len=len_trim(ref)
 do n=1,act_counter
    if (ref(1:len) == ref_counter(n)(1:len) ) then
         now(n) = now(0)
         counter(0) = counter(0)+elapsed()-now(0)
         return
    endif
 enddo
 if (act_counter == max_counter ) then
        print*,' ERROR:'
        print*,' number of counters exceeds max_counter = ',max_counter
        print*,' in tic'
        print*,' ref = ',ref
 endif
 act_counter = act_counter + 1
 ref_counter(act_counter) = ref
 now(act_counter) = elapsed()
 counter(0) = counter(0)+elapsed()-now(0)
 end subroutine tic

 subroutine toc(ref)
 implicit none
 character (len=*), intent(in) :: ref
 integer :: n,len
 now(0)=elapsed()
 len=len_trim(ref)
 do n=1,act_counter
        if (ref(1:len) == ref_counter(n)(1:len) ) then
         counter(n) = counter(n)+now(0)-now(n)
         counter(0) = counter(0)+elapsed()-now(0)
         return
        endif
 enddo
 print*,' ERROR:'
 print*,' cannot find ',ref(1:len),' in my list'
 print*,' in toc'
 end subroutine toc

 real function timing_secs(ref)
 implicit none
 character (len=*), intent(in) :: ref
 integer :: n,len
  if (ref=='tictoc') then
       timing_secs = counter(0)
       return
  endif
  len=len_trim(ref)
  do n=1,act_counter
        if (ref(1:len) == ref_counter(n)(1:len) ) then
         timing_secs = counter(n)
         return
        endif
  enddo
  timing_secs = -1.
  end function timing_secs


 subroutine get_timing(ref,secs)
 implicit none
  character (len=*), intent(in) :: ref
  real,intent(out) :: secs 
  integer :: n,len
  if (ref=='tictoc') then
       secs = counter(0)
       return
  endif
  len=len_trim(ref)
  do n=1,act_counter
        if (ref(1:len) == ref_counter(n)(1:len) ) then
         secs = counter(n)
         return
        endif
  enddo
  print*,' ERROR:'
  print*,' cannot find ',ref(1:len),' in my list'
  print*,' in get_timing'
  secs = -1.
  end subroutine



  end module timing_module
 
