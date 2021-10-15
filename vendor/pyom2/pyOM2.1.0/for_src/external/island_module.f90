
  module island_module
      logical :: cyclic
  end module

 
      subroutine isleperim(kmt,map, iperm, jperm, iofs, nippts, nisle, &
              imt, jmt, mnisle, maxipp,my_pe,cyclic_bc, verbose)
!
!=======================================================================
!         Island and Island Perimeter Mapping Routines
!         December 1993/revised February 1995
!
!=======================================================================
      use island_module
      implicit real (kind=8) (a-h,o-z)
      common /qsize/ maxqsize
      integer kmt(imt,jmt), map(imt,jmt), iperm(maxipp)
      integer jperm(maxipp), nippts(mnisle), iofs(mnisle)
      integer nisle,imt,jmt,mnisle,maxipp
      logical cyclic_bc,verbose
      parameter (maxq=10000)
      dimension iq(maxq)
      dimension jq(maxq)
      integer qfront, qback
      integer ocean,my_pe
      parameter (land=1, ocean=0)
      parameter (kmt_land=0, kmt_ocean=1)

      cyclic=cyclic_bc

      if (my_pe==0.and.verbose)   print*,' Finding perimeters of all land masses'
      if (my_pe==0.and. cyclic.and.verbose)  print*,' using cyclic boundary conditions'
!
!     initialize number of changes to kmt
!
      nchanges = 0
!    1 continue
!
!-----------------------------------------------------------------------
!     copy kmt to map changing notation
!     initially, 0 means ocean and 1 means unassigned land in map
!     as land masses are found, they are labeled 2, 3, 4, ...,
!     and their perimeter ocean cells -2, -3, -4, ...,
!     when no land points remain unassigned, land mass numbers are
!     reduced by 1 and their perimeter ocean points relabelled accordingly
!-----------------------------------------------------------------------
!
      do i=1,imt
        do j=1,jmt
          if (kmt(i,j) .gt. 0) then
            map(i,j) = ocean
          else
            map(i,j) = land
          end if
        end do
      end do

!
!-----------------------------------------------------------------------
!     find unassigned land points and expand them to continents
!-----------------------------------------------------------------------
!
      maxqsize = 0
      call qinit (iq, jq, qfront, qback)
      label = 2
      iofs(label) = 0
      nippts(label) = 0
      nerror = 0
      jnorth = jmt
      if (cyclic) then
       iwest = 2
       ieast = imt-1
      else
       iwest = 1
       ieast = imt
      endif
      do j=jnorth,1,-1
        do i=iwest,ieast
          if (map(i,j) .eq. land) then
            call qpush (i, j, iq, jq, qfront, qback)
            call expand (map, label, iq, jq, qfront, qback, nerror,iperm, jperm, iofs, nippts, &
                         imt, jmt, mnisle, maxipp)
           if (my_pe==0.and.verbose) print*,' number of island perimeter points: nippts(',label-1,')=',nippts(label)
            label = label + 1
            if (label .gt. mnisle) then
              if (my_pe==0) print '(a,i3,a)','ERROR==> mnisle=',mnisle,' is too small'
              if (my_pe==0) print*, '==> expand'
              call halt_stop(' in isleperim')
            end if
            iofs(label) = iofs(label-1) + nippts(label-1)
            nippts(label) = 0
          end if
        end do
      end do
      nisle = label - 1
!-----------------------------------------------------------------------
!     relabel land masses and their ocean perimeters
!------------------------------------------------------------------------
      do i=iwest,ieast
        do j=1,jnorth
          if (map(i,j) .ne. 0) then
            map(i,j) = map(i,j) - sign(1, map(i,j))
          end if
        end do
      end do
      do isle=2,nisle
        iofs(isle-1) = iofs(isle)
        nippts(isle-1) = nippts(isle)
      end do
      nisle = nisle - 1

      if (cyclic) then
       do j=1,jmt
         map(1,j) = map(imt-1,j)
         map(imt,j) = map(2,j)
       end do
      endif

      if (my_pe==0.and.verbose) then
       print *,' Island perimeter statistics:'
       print *,' maximum queue size was ',maxqsize
       print *,' number of land masses is ', nisle
       print *,' number of island perimeter points is ',  nippts(nisle) + iofs(nisle)
      endif

      end subroutine isleperim






      subroutine expand (map, label, iq, jq, qfront, qback, nerror,iperm, jperm, iofs, nippts, &
                         imt, jmt, mnisle, maxipp)
!-----------------------------------------------------------------------
!          The subroutine expand uses a "flood fill" algorithm
!          to expand one previously unmarked land
!          point to its entire connected land mass and its perimeter
!          ocean points.   Diagonally adjacent land points are
!          considered connected.  Perimeter "collisions" (i.e.,
!          ocean points that are adjacent to two unconnected
!          land masses) are detected and error messages generated.
!
!          The subroutine expand uses a queue of size maxq of
!          coordinate pairs of candidate points.  Suggested
!          size for maxq is 4*(imt+jmt).  Queue overflow stops
!          execution with a message to increase the size of maxq.
!          Similarly a map with more that maxipp island perimeter
!          points or more than mnisle land masses stops execution
!          with an appropriate error message.
!-----------------------------------------------------------------------
      implicit real (kind=8) (a-h,o-z)
      dimension map(imt,jmt)

      dimension iperm(maxipp)
      dimension jperm(maxipp)
      dimension nippts(mnisle)
      dimension iofs(mnisle)

      parameter (maxq=10000)
      dimension iq(maxq)
      dimension jq(maxq)
      integer qfront, qback
      logical qempty

      integer offmap, ocean
      parameter (offmap = -1)
      parameter (land = 1, ocean = 0)

      parameter (mnisle2=1000)
      logical bridge_to(1:mnisle2)
!
!      print '(a,i3)', 'Exploring land mass ',label-1
!
      if (mnisle2 .lt. mnisle) then 
        if (my_pe==0) print '(a,i4,a)','ERROR:  change parameter (mnisle2=',mnisle,') in isleperim.F'
        call halt_stop(' in isleperim')
      end if
      do isle=1,mnisle
        bridge_to(isle) = .false.
      end do
!-----------------------------------------------------------------------
!     main loop:
!        Pop a candidate point off the queue and process it.
!-----------------------------------------------------------------------
 1000 continue

      if (qempty (qfront, qback)) then
        call qinit (iq, jq, qfront, qback)
        return
      else
        call qpop (i, j, iq, jq, qfront)
!       case: (i,j) is off the map
        if (i .eq. offmap .or. j .eq. offmap) then
          goto 1000
!       case: map(i,j) is already labeled for this land mass
        else if (map(i,j) .eq. label) then
          goto 1000
!       case: map(i,j) is an ocean perimeter point of this land mass
        else if (map(i,j) .eq. -label) then
          goto 1000
!       case: map(i,j) is an unassigned land point
        else if (map(i,j) .eq. land) then
          map(i,j) = label
!         print *, 'labeling ',i,j,' as ',label
          call qpush (i,         jn_isl(j,jmt), iq, jq, qfront, qback)
          call qpush (ie_isl(i,imt), jn_isl(j,jmt), iq, jq, qfront, qback)
          call qpush (ie_isl(i,imt), j,         iq, jq, qfront, qback)
          call qpush (ie_isl(i,imt), js_isl(j), iq, jq, qfront, qback)
          call qpush (i,         js_isl(j), iq, jq, qfront, qback)
          call qpush (iw_isl(i,imt), js_isl(j), iq, jq, qfront, qback)
          call qpush (iw_isl(i,imt), j,         iq, jq, qfront, qback)
          call qpush (iw_isl(i,imt), jn_isl(j,jmt), iq, jq, qfront, qback)
          goto 1000
!       case: map(i,j) is an ocean point adjacent to this land mass
        else if (map(i,j) .eq. ocean .or. map(i,j) .lt. 0) then

!         subcase: map(i,j) is a perimeter ocean point of another mass
          if (map(i,j) .lt. 0) then
            nerror = nerror + 1
            !if (my_pe==0) print '(a,a,i3,a,i3,a,a,i3,a,i3)','PERIMETER VIOLATION==> ',&
            !         'map(',i,',',j,') is in the perimeter of both ', 'land masses ', -map(i,j)-1, ' and ', label-1
!           if we just quit processing this point here, problem points
!           will be flagged several times.
!           if we relabel them, then they are only flagged once, but
!           appear in both island perimeters, which causes problems in
!           island integrals.  current choice is quit processing.
!
!           only fill first common perimeter point detected.
!           after the first land bridge is built, subsequent collisions
!           are not problems.
!
            if (.not. bridge_to(-map(i,j)-1)) then
!             option 1: fill common perimeter point to make land bridge
!                  kmt(i,j)= 0  ! we need to declare kmt here
!                  bridge_to(-map(i,j)-1) = .true.

!              do n=1,nchanges
!                if (kmt_changes(n,1) .eq.i .and.
!     &              kmt_changes(n,2) .eq.j .and.
!     &              kmt_changes(n,4) .eq.0) then
!                  bridge_to(-map(i,j)-1) = .true.
!                end if
!              end do

            end if
            goto 1000
          end if

!         case: map(i,j) is a ocean point--label it for current mass
          map(i,j) = -label
          nippts(label) = nippts(label) + 1
!         print *, 'iofs(label)=',iofs(label)
!         print *, 'nippts(label)=',nippts(label)
          if (iofs(label) + nippts(label) .gt. maxipp) then
            if (my_pe==0) print *, 'ERROR==>  maxipp=',maxipp,' is not large enough'
            call halt_stop(' in isleperim')
          end if
          iperm(iofs(label) + nippts(label)) = i
          jperm(iofs(label) + nippts(label)) = j
          goto 1000
!       case: map(i,j) is probably labeled for another land mass
!       ************* this case should not happen **************
        else
          nerror = nerror + 1
          if (my_pe==0) print '(a,a,i3,a,i3,a,a,i3,a,i3)','ERROR ==>  ','map(',i,',',j,') is labeled for both ', &
                'land masses ', map(i,j)-1,' and ',label-1
        end if
        goto 1000
      end if
      return
      end subroutine expand


      subroutine qinit (iq, jq, qfront, qback)
      implicit real (kind=8) (a-h,o-z)
      parameter (maxq=10000)
      dimension iq(maxq)
      dimension jq(maxq)
      integer qfront, qback
      qfront = 1
      qback = 0
!     fake assignments to iq and jq to avoid "flint" warning
      iq(qfront) = 0
      jq(qfront) = 0
      return
      end subroutine qinit


      subroutine qpush (i, j, iq, jq, qfront, qback)
      implicit real (kind=8) (a-h,o-z)
      common /qsize/ maxqsize
      parameter (maxq=10000)
      dimension iq(maxq)
      dimension jq(maxq)
      integer qfront, qback
      qback = qback + 1
      if (qback .gt. maxq) then
        if (qfront .ne. 1) then
!         shift queue left to make room
          ishift = qfront - 1
          do ip=qfront,qback-1
            iq(ip-ishift) = iq(ip)
            jq(ip-ishift) = jq(ip)
          end do
          qfront = 1
          qback = qback - ishift
        else
          call halt_stop( 'queue fault in qpush')
        end if
      end if
      iq(qback) = i
      jq(qback) = j
      maxqsize = max(maxqsize, (qback-qfront))
      return
      end subroutine qpush


      subroutine qpop (i, j, iq, jq, qfront)
      implicit real (kind=8) (a-h,o-z)
      parameter (maxq=10000)
      dimension iq(maxq)
      dimension jq(maxq)
      integer qfront
      i = iq(qfront)
      j = jq(qfront)
      qfront = qfront + 1
      return
      end subroutine qpop


      function qempty (qfront, qback)
      implicit real (kind=8) (a-h,o-z)
      parameter (maxq=10000)
      integer qfront, qback
      logical qempty
      qempty = (qfront .gt. qback)
      return
      end function qempty



      function jn_isl(j,jmt)
      implicit real (kind=8) (a-h,o-z)
!     j coordinate to the north of j
      integer offmap
      parameter (offmap = -1)
      if (j .lt. jmt) then
        jn_isl = j + 1
      else
        jn_isl = offmap
      end if
      return
      end function jn_isl


      function js_isl(j)
      implicit real (kind=8) (a-h,o-z)
!     j coordinate to the south of j
      integer offmap
      parameter (offmap = -1)
      if (j .gt. 1) then
        js_isl = j - 1
      else
        js_isl = offmap
      end if
      return
      end function js_isl



      function ie_isl(i,imt)
      use island_module
      implicit real (kind=8) (a-h,o-z)
!     i coordinate to the east of i
      integer offmap
      parameter (offmap = -1)

      if (cyclic) then
       if (i .lt. imt-1) then
         ie_isl = i + 1
       else
         ie_isl = (i+1) - imt + 2
       end if
      else
       if (i .lt. imt) then
        ie_isl = i + 1
       else
        ie_isl = offmap
       end if
      end if
      return
      end function ie_isl



      function iw_isl(i,imt)
      use island_module
      implicit real (kind=8) (a-h,o-z)
!     i coordinate to the west of i
      integer offmap
      parameter (offmap = -1)

      if (cyclic) then
       if (i .gt. 2) then
        iw_isl = i - 1
       else
        iw_isl = (i-1) + imt - 2
       end if
      else
       if (i .gt. 1) then
        iw_isl = i - 1
       else
        iw_isl = offmap
       end if
      end if
      return
      end function iw_isl



