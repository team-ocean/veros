import sys
import numpy as np
import Queue

def isleperim(kmt, Map, iperm, jperm, iofs, nippts, nisle, imt, jmt, mnisle, maxipp,my_pe,cyclic_bc, verbose):
    """
    =======================================================================
             Island and Island Perimeter Mapping Routines
             December 1993/revised February 1995

    =======================================================================
    """
    #use island_module
    #implicit real (kind=8) (a-h,o-z)
    #common /qsize/ maxqsize
    #integer kmt(imt,jmt), map(imt,jmt), iperm(maxipp)
    #integer jperm(maxipp), nippts(mnisle), iofs(mnisle)
    #integer nisle,imt,jmt,mnisle,maxipp
    #logical cyclic_bc,verbose
    #parameter (maxq=10000)
    #dimension iq(maxq)
    #dimension jq(maxq)
    #integer qfront, qback
    #integer ocean,my_pe
    #parameter (land=1, ocean=0)
    #parameter (kmt_land=0, kmt_ocean=1)

    cyclic = cyclic_bc

    if my_pe == 0 and verbose:
        print ' Finding perimeters of all land masses'
    if my_pe == 0 and  cyclic and verbose:
        print ' using cyclic boundary conditions'

    # initialize number of changes to kmt

    nchanges = 0
    """
    -----------------------------------------------------------------------
         copy kmt to map changing notation
         initially, 0 means ocean and 1 means unassigned land in map
         as land masses are found, they are labeled 2, 3, 4, ...,
         and their perimeter ocean cells -2, -3, -4, ...,
         when no land points remain unassigned, land mass numbers are
         reduced by 1 and their perimeter ocean points relabelled accordingly
    -----------------------------------------------------------------------
    """

    for i in xrange(1, imt): #i=1,imt
        for j in xrange(1, jmt): #j=1,jmt
            if kmt[i,j] > 0:
                Map[i,j] = ocean
            else:
                Map[i,j] = land


    """
    -----------------------------------------------------------------------
         find unassigned land points and expand them to continents
    -----------------------------------------------------------------------
    """
    maxqsize = 0
    queue = Queue.Queue()
    label = 2
    iofs[label] = 0
    nippts[label] = 0
    nerror = 0
    jnorth = jmt
    if cyclic:
        iwest = 2
        ieast = imt-1
    else:
        iwest = 1
        ieast = imt
    for j in xrange(jnorth, 0, -1): #j=jnorth,1,-1
        for i in xrange(iwest, ieast+1): #i=iwest,ieast
            if Map[i,j] == land:
                queue.put((i,j))
                expand (Map, label, queue, nerror,iperm, jperm, iofs, nippts, imt, jmt, mnisle, maxipp)
                if my_pe == 0 and verbose:
                    print ' number of island perimeter points: nippts(',label-1,')=',nippts[label]
                label = label + 1
                if label > mnisle:
                    if my_pe == 0:
                        print 'ERROR==> mnisle=',mnisle,' is too small'
                    if my_pe == 0:
                        print '==> expand'
                    sys.exit(' in isleperim')
                iofs[label] = iofs[label-1] + nippts[label-1]
                nippts[label] = 0
    nisle = label - 1
    """
    -----------------------------------------------------------------------
         relabel land masses and their ocean perimeters
    ------------------------------------------------------------------------
    """
    for i in xrange(iwest, ieast+1): #i=iwest,ieast
        for j in xrange(1,jnorth+1): #j=1,jnorth
            if Map[i,j] != 0:
                Map[i,j] -= np.sign(Map[i,j])
    for isle in xrange(2, nisle+1): #isle=2,nisle
        iofs[isle-1] = iofs[isle]
        nippts[isle-1] = nippts[isle]
    nisle = nisle - 1

    if cyclic:
        for j in xrange(1, jmt+1): #j=1,jmt
            Map[1,j] = Map[imt-1,j]
            Map[imt,j] = Map[2,j]

    if my_pe == 0 and verbose:
        print ' Island perimeter statistics:'
        print ' maximum queue size was ',maxqsize
        print ' number of land masses is ', nisle
        print ' number of island perimeter points is ',  nippts[nisle] + iofs[nisle]

def expand(Map, label, queue, nerror,iperm, jperm, iofs, nippts, imt, jmt, mnisle, maxipp):
    """
    -----------------------------------------------------------------------
              The subroutine expand uses a "flood fill" algorithm
              to expand one previously unmarked land
              point to its entire connected land mass and its perimeter
              ocean points.   Diagonally adjacent land points are
              considered connected.  Perimeter "collisions" (i.e.,
              ocean points that are adjacent to two unconnected
              land masses) are detected and error messages generated.

              The subroutine expand uses a queue of size maxq of
              coordinate pairs of candidate points.  Suggested
              size for maxq is 4*(imt+jmt).  Queue overflow stops
              execution with a message to increase the size of maxq.
              Similarly a map with more that maxipp island perimeter
              points or more than mnisle land masses stops execution
              with an appropriate error message.
    -----------------------------------------------------------------------
    """
    #implicit real (kind=8) (a-h,o-z)
    #dimension map(imt,jmt)

    #dimension iperm(maxipp)
    #dimension jperm(maxipp)
    #dimension nippts(mnisle)
    #dimension iofs(mnisle)

    #parameter (maxq=10000)
    #dimension iq(maxq)
    #dimension jq(maxq)
    #integer qfront, qback
    #logical qempty

    #integer offmap, ocean
    #parameter (offmap = -1)
    #parameter (land = 1, ocean = 0)

    #parameter (mnisle2=1000)
    #logical bridge_to(1:mnisle2)

    if mnisle2 < mnisle:
        if my_pe == 0:
            print 'ERROR:  change parameter (mnisle2=',mnisle,') in isleperim.F'
        sys.exit(' in isleperim')
    for isle in xrange(1, mnisle): #isle=1,mnisle
        bridge_to[isle] = False
    """
    -----------------------------------------------------------------------
         main loop:
            Pop a candidate point off the queue and process it.
    -----------------------------------------------------------------------
    """
    while not queue.empty():
        (i,j) = queue.get()
#       case: (i,j) is off the map
        if i == offmap or j == offmap:
            continue
#       case: map(i,j) is already labeled for this land mass
        elif Map[i,j] == label:
            continue
#       case: map(i,j) is an ocean perimeter point of this land mass
        elif Map[i,j] == -label:
            continue
#       case: map(i,j) is an unassigned land point
        elif Map[i,j] == land:
            Map[i,j] = label
#           print *, 'labeling ',i,j,' as ',label
            queue.put((i, jn_isl[j,jmt]))
            queue.put((ie_isl[i,imt], jn_isl[j,jmt]))
            queue.put((ie_isl[i,imt], j))
            queue.put((ie_isl[i,imt], js_isl[j]))
            queue.put((i, js_isl[j]))
            queue.put((iw_isl[i,imt], js_isl[j]))
            queue.put((iw_isl[i,imt], j))
            queue.put((iw_isl[i,imt], jn_isl[j,jmt]))
#       case: map(i,j) is an ocean point adjacent to this land mass
        elif Map[i,j] == ocean or Map[i,j] < 0:

#           subcase: map(i,j) is a perimeter ocean point of another mass
            if Map[i,j] < 0:
                nerror = nerror + 1
                #if (my_pe==0) print '(a,a,i3,a,i3,a,a,i3,a,i3)','PERIMETER VIOLATION==> ',&
                #         'map(',i,',',j,') is in the perimeter of both ', 'land masses ', -map(i,j)-1, ' and ', label-1
#               if we just quit processing this point here, problem points
#               will be flagged several times.
#               if we relabel them, then they are only flagged once, but
#               appear in both island perimeters, which causes problems in
#               island integrals.  current choice is quit processing.
#
#               only fill first common perimeter point detected.
#               after the first land bridge is built, subsequent collisions
#               are not problems.
#
#                if (.not. bridge_to(-map(i,j)-1)) then
#                 option 1: fill common perimeter point to make land bridge
#                      kmt(i,j)= 0  ! we need to declare kmt here
#                      bridge_to(-map(i,j)-1) = .true.

#                  do n=1,nchanges
#                    if (kmt_changes(n,1) .eq.i .and.
#     &                  kmt_changes(n,2) .eq.j .and.
#     &                  kmt_changes(n,4) .eq.0) then
#                      bridge_to(-map(i,j)-1) = .true.
#                    end if
#                  end do
#
#                end if
                continue

#           case: map(i,j) is a ocean point--label it for current mass
            Map[i,j] = -label
            nippts[label] = nippts[label] + 1
#           print *, 'iofs(label)=',iofs(label)
#           print *, 'nippts(label)=',nippts(label)
            if iofs[label] + nippts[label] > maxipp:
                if my_pe == 0:
                    print 'ERROR==>  maxipp=',maxipp,' is not large enough'
                sys.exit(' in isleperim')
            iperm[iofs[label] + nippts[label]] = i
            jperm[iofs[label] + nippts[label]] = j
#       case: map(i,j) is probably labeled for another land mass
#       ************* this case should not happen **************
        else:
            nerror = nerror + 1
            if my_pe == 0:
                print 'ERROR ==>  ','map(',i,',',j,') is labeled for both ', 'land masses ', Map[i,j]-1,' and ',label-1
        return


def jn_isl(j,jmt):
#     j coordinate to the north of j
      #integer offmap
    offmap = -1
    if j < jmt:
      return j + 1
    else:
        return offmap


def js_isl(j):
#      implicit real (kind=8) (a-h,o-z)
#     j coordinate to the south of j
#      integer offmap
#      parameter (offmap = -1)
    offmap = -1
    if j > 1:
        return j - 1
    else:
        return offmap


def ie_isl(i, imt):
#     use island_module
#     implicit real (kind=8) (a-h,o-z)
#     i coordinate to the east of i
#     integer offmap
#     parameter (offmap = -1)
    offmap = -1
    if cyclic:
        if i < imt-1:
            return i+1
        else:
            return i+1 - imt + 2
    else:
        if i < imt:
            return i + 1
        else:
            return offmap


def iw_isl(i, imt):
    offmap = -1
    if cyclic:
        if i > 2:
            return i-1
        else:
            return i-1 + imt - 2
    else:
        if i > 1:
            return i - 1
        else:
            return offmap
