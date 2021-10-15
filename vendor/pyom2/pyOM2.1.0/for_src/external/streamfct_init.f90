





subroutine streamfunction_init
!=======================================================================
!  prepare for island integrals
!=======================================================================
 use main_module   
 implicit none
 integer :: allmap(1-onx:nx+onx,1-onx:ny+onx)
 integer :: map(1-onx:nx+onx,1-onx:ny+onx)
 integer :: kmt(1-onx:nx+onx,1-onx:ny+onx)
 integer, parameter :: maxipp = 10000, mnisle = 1000
 integer :: iperm(maxipp),jperm(maxipp),nippts(mnisle), iofs(mnisle)
 integer :: isle,n,i,j,ij(2),max_boundary,dir(2),ijp(2),ijp_right(2)
 logical :: cont,verbose ,converged
 real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 real*8 :: fpx(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 real*8 :: fpy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)

 if (my_pe==0) print'(/,a,/)','Initializing streamfunction method'
 verbose = enable_congrad_verbose
 !-----------------------------------------------------------------------
 ! communicate kbot to get the entire land map
 !-----------------------------------------------------------------------
 kmt=0; ! note that routine will modify kmt
 do j=js_pe,je_pe
   do i=is_pe,ie_pe
     if (kbot(i,j)>0) kmt(i,j)=5
   enddo
 enddo
 call pe0_recv_2D_int(nx,ny,kmt(1:nx,1:ny))
 call pe0_bcast_int(kmt,(nx+2*onx)*(ny+2*onx))
 if (enable_cyclic_x) then
    do i=1,onx
       kmt(nx+i,:)=kmt(i  ,:)
       kmt(1-i,:) =kmt(nx-i+1,:) 
    enddo
 endif

 !-----------------------------------------------------------------------
 ! preprocess land map using MOMs algorithm for B-grid to determine number of islands
 !-----------------------------------------------------------------------
 if (my_pe==0) print'(a)',' starting MOMs algorithm for B-grid to determine number of islands'
 call isleperim(kmt,allmap, iperm, jperm, iofs, nippts, nisle, &
                nx+2*onx, ny+2*onx, mnisle, maxipp,my_pe,enable_cyclic_x,.true.)
 if (enable_cyclic_x) then
      do i=1,onx
       allmap(nx+i,:)=allmap(i  ,:)
       allmap(1-i,:) =allmap(nx-i+1,:) 
      enddo
 endif
 if (my_pe==0) call showmap(1-onx,nx+onx,1-onx,ny+onx,  allmap)

 !-----------------------------------------------------------------------
 ! allocate variables
 !-----------------------------------------------------------------------
 max_boundary= 2*maxval(nippts(1:nisle))
 allocate( boundary(nisle,max_boundary,2)); boundary=0
 allocate( line_dir(nisle,max_boundary,2)); line_dir = 0
 allocate( nr_boundary(nisle)  ) ; nr_boundary=0
 allocate( psin(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,nisle) ); psin=0.
 allocate( dpsin(nisle,3) ); dpsin=0.
 allocate( line_psin(nisle,nisle) ); line_psin=0.

 do isle=1,nisle

   if (my_pe==0) then
     print'(/,a)',' ------------------------'
     print'(a,i8)',' processing island #',isle
     print'(a,/)',' ------------------------'
   endif

   !-----------------------------------------------------------------------
   ! land map for island number isle: 1 is land, -1 is perimeter, 0 is ocean
   !-----------------------------------------------------------------------
   kmt=1; where (allmap==isle) kmt=0
   call isleperim(kmt,map, iperm, jperm, iofs, nippts, i, &
                  nx+2*onx, ny+2*onx, mnisle, maxipp,my_pe,enable_cyclic_x,.false.)
   if (my_pe==0 .and. verbose) call showmap(1-onx,nx+onx,1-onx,ny+onx, map)

   !-----------------------------------------------------------------------
   ! find a starting point   
   !-----------------------------------------------------------------------
   n=1
   cont = .false.
   outerloop: do i=nx/2+1,nx  ! avoid starting close to cyclic bondaries
    do j=0,ny
       if (map(i,j)==1 .and.  map(i,j+1)==-1 ) then
           ! initial direction is eastward, we come from the west
           ij=(/i,j/); cont = .true.
           dir = (/ 1,0/)
           boundary(isle,n,:)= (/ ij(1)-1,ij(2)/)
           exit outerloop
       endif
       if (map(i,j)==-1 .and.  map(i,j+1)==1 ) then
           ! initial direction is westward, we come from the east
           ij=(/i-1,j/); cont = .true.
           dir = (/ -1,0/)
           boundary(isle,n,:)= (/ ij(1)+1,ij(2)/)
           exit outerloop
       endif
    enddo
   enddo outerloop

   if ( .not.cont ) then
    outerloop2: do i=nx/2,1,-1  ! avoid starting close to cyclic bondaries
     do j=0,ny
       if (map(i,j)==1 .and.  map(i,j+1)==-1 ) then
           ! initial direction is eastward, we come from the west
           ij=(/i,j/); cont = .true.
           dir = (/ 1,0/)
           boundary(isle,n,:)= (/ ij(1)-1,ij(2)/)
           exit outerloop2
       endif
       if (map(i,j)==-1 .and.  map(i,j+1)==1 ) then
           ! initial direction is westward, we come from the east
           ij=(/i-1,j/); cont = .true.
           dir = (/ -1,0/)
           boundary(isle,n,:)= (/ ij(1)+1,ij(2)/)
           exit outerloop2
       endif
     enddo
    enddo outerloop2

    if ( .not.cont ) then
       if (my_pe==0) print'(a)','found no starting point for line integral'
       call halt_stop('in streamfunction_init')
    endif

   endif

   if (my_pe==0) then 
          print'(a,2i6)',' starting point of line integral is ',boundary(isle,n,:)
          print'(a,2i6)',' starting direction is ',dir
   endif

   !-----------------------------------------------------------------------
   ! now find connecting lines
   !-----------------------------------------------------------------------
   line_dir(isle,n,:)= dir
   n=2
   boundary(isle,n,:)= (/ ij(1),ij(2)/)
   cont = .true.
   do while (cont) 
     !-----------------------------------------------------------------------
     ! consider map in front of line direction and to the right and decide where to go
     !-----------------------------------------------------------------------
     if (dir(1)== 0.and.dir(2)== 1) ijp      =(/ij(1)  ,ij(2)+1/) !north
     if (dir(1)== 0.and.dir(2)== 1) ijp_right=(/ij(1)+1,ij(2)+1/) !north
     if (dir(1)==-1.and.dir(2)== 0) ijp      =(/ij(1)  ,ij(2)  /) !west
     if (dir(1)==-1.and.dir(2)== 0) ijp_right=(/ij(1)  ,ij(2)+1/) !west
     if (dir(1)== 0.and.dir(2)==-1) ijp      =(/ij(1)+1,ij(2)  /) !south
     if (dir(1)== 0.and.dir(2)==-1) ijp_right=(/ij(1)  ,ij(2)  /) !south
     if (dir(1)== 1.and.dir(2)== 0) ijp      =(/ij(1)+1,ij(2)+1/) !east
     if (dir(1)== 1.and.dir(2)== 0) ijp_right=(/ij(1)+1,ij(2)  /) !east

     !-----------------------------------------------------------------------
     !  4 cases are possible
     !-----------------------------------------------------------------------

     if (verbose .and. my_pe==0) then
      print*,' ' 
      print*,' position is  ',ij
      print*,' direction is ',dir
      print*,' map ahead is ',map(ijp(1),ijp(2)) , map(ijp_right(1),ijp_right(2)) 
     endif

     if (map(ijp(1),ijp(2))==-1 .and. map(ijp_right(1),ijp_right(2)) == 1 ) then
       if (verbose .and. my_pe==0) print*,' go forward'
     else if (map(ijp(1),ijp(2))==-1 .and. map(ijp_right(1),ijp_right(2)) == -1 ) then
       if (verbose .and. my_pe==0) print*,' turn right'
       dir = (/dir(2),-dir(1)/)
     else if (map(ijp(1),ijp(2))== 1 .and. map(ijp_right(1),ijp_right(2)) == 1 ) then
       if (verbose .and. my_pe==0) print*,' turn left'
       dir =  (/-dir(2),dir(1)/) 
     else if (map(ijp(1),ijp(2))== 1 .and. map(ijp_right(1),ijp_right(2)) == -1 ) then
       if (verbose .and. my_pe==0) print*,' turn left'
       dir =  (/-dir(2),dir(1)/) 
     else
      print'(a)','unknown situation or lost track'
      do n=1,n
       print*,' pos=',boundary(isle,n,:),' dir=',line_dir(isle,n,:)
      enddo
      print*,' map ahead is ',map(ijp(1),ijp(2)) , map(ijp_right(1),ijp_right(2)) 
      call halt_stop(' in streamfunction_init ')
     endif

     !-----------------------------------------------------------------------
     ! go forward in direction
     !-----------------------------------------------------------------------
     line_dir(isle,n,:)= dir
     ij = ij + dir
     if (boundary(isle,1,1)==ij(1).and.boundary(isle,1,2)==ij(2) )  cont=.false.

     !-----------------------------------------------------------------------
     ! account for cyclic boundary conditions
     !-----------------------------------------------------------------------
     if (enable_cyclic_x.and.dir(1)== 1.and.dir(2)== 0.and.ij(1)>nx) then
         if (verbose .and. my_pe==0)  print*,' shifting to western cyclic boundary'
         ij(1)=ij(1)-nx
     endif
     if (enable_cyclic_x.and.dir(1)==-1.and.dir(2)== 0.and.ij(1)<1) then
         if (verbose .and. my_pe==0)  print*,' shifting to eastern cyclic boundary'
         ij(1)=ij(1)+nx
     endif
     if (boundary(isle,1,1)==ij(1).and.boundary(isle,1,2)==ij(2) )  cont=.false.

     if (cont) then
       n=n+1
       if (n>max_boundary) then
         print'(a)','increase value of max_boundary'
         call halt_stop(' in streamfunction_init ')
       endif
       boundary(isle,n,:)= ij
     endif 

   enddo
   nr_boundary(isle)=n
   if (my_pe==0) print'(a,i8)',' number of points is ',n
   if (verbose .and. my_pe==0) then
    print*,' '
    print*,' Positions:'
    do n=1,nr_boundary(isle)
      print*,' pos=',boundary(isle,n,:),' dir=',line_dir(isle,n,:)
    enddo
   endif
 enddo

 !-----------------------------------------------------------------------
 ! precalculate time independent boundary components of streamfunction
 !-----------------------------------------------------------------------
 forc=0.0
 do isle=1,nisle
   psin(:,:,isle)=0.0
   do n=1,nr_boundary(isle)
     i=boundary(isle,n,1)
     j=boundary(isle,n,2)
     if ( i>=is_pe-onx .and. i<=ie_pe+onx .and. j>=js_pe-onx .and. j<=je_pe+onx ) psin(i,j,isle)=1.0
   enddo
   call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,psin(:,:,isle)); 
   call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,psin(:,:,isle))
   if (my_pe==0) print'(a,i4)',' solving for boundary contribution by island ',isle
   call congrad_streamfunction(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,forc,congr_itts,psin(:,:,isle),converged)
   if (my_pe==0) print'(a,i8)',' itts =  ',congr_itts
   call border_exchg_xy(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,psin(:,:,isle)); 
   call setcyclic_xy   (is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx,psin(:,:,isle))
 enddo

 !-----------------------------------------------------------------------
 ! precalculate time independent island integrals
 !-----------------------------------------------------------------------
 do n=1,nisle
   do isle=1,nisle
    fpx=0;fpy=0
    do j=js_pe-onx+1,je_pe+onx
     do i=is_pe-onx+1,ie_pe+onx
      fpx(i,j) =-maskU(i,j,nz)*( psin(i,j,isle)-psin(i,j-1,isle))/dyt(j)*hur(i,j)
      fpy(i,j) = maskV(i,j,nz)*( psin(i,j,isle)-psin(i-1,j,isle))/(cosu(j)*dxt(i))*hvr(i,j)
     enddo
    enddo
    call line_integral(is_pe-onx,ie_pe+onx,js_pe-onx,je_pe+onx, n,fpx,fpy,line_psin(n,isle))
   enddo
 enddo
end subroutine






subroutine line_integral(is_,ie_,js_,je_,isle,uloc,vloc,line)
!=======================================================================
! calculate line integral along island isle
!=======================================================================
 use main_module
 implicit none
 integer :: isle
 integer :: js_,je_,is_,ie_
 !real*8 :: uloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 !real*8 :: vloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
 real*8 :: uloc(is_:ie_,js_:je_)
 real*8 :: vloc(is_:ie_,js_:je_)
 real*8 :: line
 integer :: n,i,j,nm1,js,je,is,ie
   line = 0
   is = is_pe; js = js_pe; ie = ie_pe; je = je_pe
   if (is ==1 ) is=is-1
   if (js ==1 ) js=js-1

   do n=1,nr_boundary(isle)
    nm1 = n-1; if (nm1<1) nm1=nr_boundary(isle)
    i=boundary(isle,n,1);
    j=boundary(isle,n,2)
    if ( i>=is .and. i<=ie .and. j>=js .and. j<=je ) then  

     if     (line_dir(isle,n,1) ==  1 .and. line_dir(isle,n,2) ==  0) then   ! to east

       !if      (line_dir(isle,nm1,1) ==  1 .and. line_dir(isle,nm1,2) ==  0) then ! from west, straight
       !    line = line + uloc(i,j+1)*dxu(i)*cost(j+1)
       !else if (line_dir(isle,nm1,1) ==  0 .and. line_dir(isle,nm1,2) ==  1) then ! from south, outer turn
           line = line + vloc(i,j)*dyu(j) + uloc(i,j+1)*dxu(i)*cost(j+1)
       !else if (line_dir(isle,nm1,1) ==  0 .and. line_dir(isle,nm1,2) == -1) then ! from north, inner turn, no contribution
       !endif

     else if (line_dir(isle,n,1) == -1 .and. line_dir(isle,n,2) ==  0) then ! to west

       !if      (line_dir(isle,nm1,1) == -1 .and. line_dir(isle,nm1,2) ==  0) then ! from east, straight
       !   line = line - uloc(i,j)*dxu(i)*cost(j)
       !else if (line_dir(isle,nm1,1) ==  0 .and. line_dir(isle,nm1,2) == -1) then ! from north, outer turn
           line = line - vloc(i+1,j)*dyu(j) - uloc(i,j)*dxu(i)*cost(j)
       !else if (line_dir(isle,nm1,1) ==  0 .and. line_dir(isle,nm1,2) ==  1) then ! from south, inner turn, no contribution
       !endif

     else if (line_dir(isle,n,1) ==  0 .and. line_dir(isle,n,2) ==  1) then  ! to north

       !if      (line_dir(isle,nm1,1) ==  0 .and. line_dir(isle,nm1,2) ==  1) then ! from south, straight
       !    line = line + vloc(i,j)*dyu(j)
       !else if (line_dir(isle,nm1,1) == -1 .and. line_dir(isle,nm1,2) == 0 ) then ! from east, outer turn
           line = line + vloc(i,j)*dyu(j)  - uloc(i,j)*dxu(i)*cost(j)
       !else if (line_dir(isle,nm1,1) ==  1 .and. line_dir(isle,nm1,2) ==  0) then ! from west, inner turn, no contrib.
       !endif

     else if (line_dir(isle,n,1) ==  0 .and. line_dir(isle,n,2) == -1) then  ! to south

       !if      (line_dir(isle,nm1,1) ==  0 .and. line_dir(isle,nm1,2) == -1) then ! from north, straight
       !     line = line - vloc(i+1,j)*dyu(j)
       !else if (line_dir(isle,nm1,1) ==  1 .and. line_dir(isle,nm1,2) ==  0) then ! from west, outer turn
           line = line + uloc(i,j+1)*dxu(i)*cost(j+1) - vloc(i+1,j)*dyu(j)
       !else if (line_dir(isle,nm1,1) == -1 .and. line_dir(isle,nm1,2) == 0 ) then ! from east, inner turn, no contrib
       !endif

     else
       print*,' line_dir =',line_dir(isle,n,:),' at pos. ',boundary(isle,n,:)
       call halt_stop(' missing line_dir in line integral')
     endif
    endif
   enddo
   call global_sum(line)
end subroutine



integer function mod10(m)
 implicit none
 integer :: m
 if (m .eq. 0) then
    mod10 = 0
 else if (m .gt. 0) then
     mod10 = mod(m,10)
 else
     mod10 = m
 end if
end function 


subroutine showmap (is_,ie_,js_,je_,map)
 use main_module
 implicit none
 integer :: js_,je_,is_,ie_
 !integer :: map(1-onx:nx+onx,1-onx:ny+onx)
 integer :: map(is_:ie_,js_:je_)
 integer,parameter :: linewidth=125
 integer :: istart,iremain,isweep,iline,i,j,mod10
 integer :: imt

 imt=nx +2*onx
 iremain = imt
 istart = 0
 print '(/,132a)',(' ',i=1,5+min(linewidth,imt)/2-13),'Land mass and perimeter'
 do isweep=1,imt/linewidth + 1
  iline = min(iremain, linewidth)
  iremain = iremain - iline
  if (iline .gt. 0) then
        print *, ' '
        print '(t3,32i5)', (istart+i+1-onx,i=1,iline,5)
        do j=ny+onx,1-onx,-1
            print '(i4,t6,160i1)', j,(mod10(map(istart+i -onx,j)),i=1,iline)
        end do
        print '(t3,32i5)', (istart+i+1-onx,i=1,iline,5)
        !print '(t6,32i5)', (istart+i+4-onx,i=1,iline,5)
        istart = istart + iline
  end if
 end do
 print *, ' '
end subroutine showmap



