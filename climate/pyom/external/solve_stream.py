"""
=======================================================================
      solve two dimensional Possion equation
           A * dpsi = forc,  where A = nabla_h^2
      with Dirichlet boundary conditions
      used for streamfunction
=======================================================================
"""

import sys
import numpy as np
from scipy.linalg import lapack
import climate
from climate.pyom.external import solve_pressure, island
from climate.pyom import cyclic

def streamfunction_init(pyom):
    """
    =======================================================================
      prepare for island integrals
    =======================================================================
    """
    #integer :: allmap(1-onx:nx+onx,1-onx:ny+onx)
    #integer :: map(1-onx:nx+onx,1-onx:ny+onx)
    #integer :: kmt(1-onx:nx+onx,1-onx:ny+onx)
    #integer :: iperm(maxipp),jperm(maxipp),nippts(mnisle), iofs(mnisle)
    #integer :: isle,n,i,j,ij(2),max_boundary,dir(2),ijp(2),ijp_right(2)
    #logical :: cont,verbose ,converged
    #real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: fpx(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: fpy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    maxipp = 10000
    mnisle = 1000
    allmap = np.zeros((pyom.nx+4, pyom.ny+4))
    Map    = np.zeros((pyom.nx+4, pyom.ny+4))
    forc   = np.zeros((pyom.nx+4, pyom.ny+4))
    fpx    = np.empty((pyom.nx+4, pyom.ny+4))
    fpy    = np.empty((pyom.nx+4, pyom.ny+4))
    iperm  = np.zeros(maxipp)
    jperm  = np.zeros(maxipp)
    nippts = np.zeros(mnisle, dtype=np.int)
    iofs   = np.zeros(mnisle, dtype=np.int)

    if climate.is_bohrium:
        allmap = allmap.copy2numpy()
        Map = Map.copy2numpy()

    print 'Initializing streamfunction method'
    verbose = pyom.enable_congrad_verbose
    """
    -----------------------------------------------------------------------
     communicate kbot to get the entire land map
    -----------------------------------------------------------------------
    """
    kmt = np.zeros((pyom.nx+4, pyom.ny+4)) # note that routine will modify kmt
    if climate.is_bohrium:
        kbot = pyom.kbot.copy2numpy()
        kmt = kmt.copy2numpy()
    else:
        kbot = pyom.kbot
    kmt[2:pyom.nx+2, 2:pyom.ny+2][kbot[2:pyom.nx+2, 2:pyom.ny+2] > 0] = 5

    #MPI stuff
    #call pe0_recv_2D_int(nx,ny,kmt(1:nx,1:ny))
    #call pe0_bcast_int(kmt,(nx+2*onx)*(ny+2*onx))

    if pyom.enable_cyclic_x:
        for i in xrange(1, 3): #i=1,onx
            kmt[pyom.nx+i+1,:] = kmt[i+1  ,:]
            kmt[2-i,:] = kmt[pyom.nx-i+2,:]

    """
    -----------------------------------------------------------------------
     preprocess land map using MOMs algorithm for B-grid to determine number of islands
    -----------------------------------------------------------------------
    """
    print ' starting MOMs algorithm for B-grid to determine number of islands'
    island.isleperim(kmt,allmap, iperm, jperm, iofs, nippts, pyom.nx+4, pyom.ny+4, mnisle, maxipp,pyom,change_nisle=True, verbose=True)
    if pyom.enable_cyclic_x:
        for i in xrange(1, 3): #i=1,onx
            allmap[pyom.nx+i+1,:] = allmap[i+1  ,:]
            allmap[2-i,:] = allmap[pyom.nx-i+2,:]
    showmap(allmap, pyom)

    """
    -----------------------------------------------------------------------
     allocate variables
    -----------------------------------------------------------------------
    """
    max_boundary= 2*np.max(nippts[:pyom.nisle])
    pyom.boundary = np.zeros((pyom.nisle, max_boundary, 2), dtype=np.int)
    pyom.line_dir = np.zeros((pyom.nisle, max_boundary, 2))
    pyom.nr_boundary = np.zeros(pyom.nisle, dtype=np.int)
    pyom.psin = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nisle))
    pyom.dpsin = np.zeros((pyom.nisle, 3))
    pyom.line_psin = np.zeros((pyom.nisle, pyom.nisle))

    for isle in xrange(pyom.nisle): #isle=1,nisle

        print ' ------------------------'
        print ' processing island #',isle
        print ' ------------------------'

        """
        -----------------------------------------------------------------------
         land map for island number isle: 1 is land, -1 is perimeter, 0 is ocean
        -----------------------------------------------------------------------
        """
        kmt[...] = 1
        kmt[allmap == isle+1] = 0
        island.isleperim(kmt,Map, iperm, jperm, iofs, nippts, pyom.nx+4, pyom.ny+4, mnisle, maxipp,pyom)
        if verbose:
            showmap(Map, pyom)

        """
        -----------------------------------------------------------------------
         find a starting point
        -----------------------------------------------------------------------
        """
        n=0
        # avoid starting close to cyclic bondaries
        (cont, ij, Dir) = avoid_cyclic_boundaries(Map, isle, n, pyom)

        if not cont:
            (cont, ij, Dir) = avoid_cyclic_boundaries2(Map, isle, n, pyom)

            if not cont:
                raise RuntimeError('found no starting point for line integral')

        print ' starting point of line integral is ',pyom.boundary[isle,n,:]
        print ' starting direction is ', Dir

        """
        -----------------------------------------------------------------------
         now find connecting lines
        -----------------------------------------------------------------------
        """
        pyom.line_dir[isle,n,:] = Dir
        n = 1
        pyom.boundary[isle,n,:] = np.array([ij[0], ij[1]])
        cont = True
        while cont:
            """
            -----------------------------------------------------------------------
             consider map in front of line direction and to the right and decide where to go
            -----------------------------------------------------------------------
            """
            if Dir[0]== 0 and Dir[1]== 1:
                ijp      = [ij[0]  ,ij[1]+1] #north
                ijp_right= [ij[0]+1,ij[1]+1] #north
            if Dir[0]==-1 and Dir[1]== 0:
                ijp      = [ij[0]  ,ij[1]  ] #west
                ijp_right= [ij[0]  ,ij[1]+1] #west
            if Dir[0]== 0 and Dir[1]==-1:
                ijp      = [ij[0]+1,ij[1]  ] #south
                ijp_right= [ij[0]  ,ij[1]  ] #south
            if Dir[0]== 1 and Dir[1]== 0:
                ijp      = [ij[0]+1,ij[1]+1] #east
                ijp_right= [ij[0]+1,ij[1]  ] #east

            """
            -----------------------------------------------------------------------
              4 cases are possible
            -----------------------------------------------------------------------
            """

            if verbose:
                print ' '
                print ' position is  ',ij
                print ' direction is ',Dir
                print ' map ahead is ',Map[ijp[0],ijp[1]] , Map[ijp_right[0],ijp_right[1]]

            if Map[ijp[0],ijp[1]] == -1 and Map[ijp_right[0],ijp_right[1]] == 1:
                if verbose:
                    print ' go forward'
            elif Map[ijp[0],ijp[1]] == -1 and Map[ijp_right[0],ijp_right[1]] == -1:
                if verbose:
                    print ' turn right'
                Dir = np.array([Dir[1],-Dir[0]])
            elif Map[ijp[0],ijp[1]] == 1 and Map[ijp_right[0],ijp_right[1]] == 1:
                if verbose:
                    print ' turn left'
                Dir =  np.array([-Dir[1],Dir[0]])
            elif Map[ijp[0],ijp[1]] == 1 and Map[ijp_right[0],ijp_right[1]] == -1:
                if verbose:
                    print ' turn left'
                Dir =  np.array([-Dir[1],Dir[0]])
            else:
                print 'unknown situation or lost track'
                for n in xrange(1, n+1): #n=1,n
                    print ' pos=',pyom.boundary[isle,n,:],' dir=',line_dir[isle,n,:]
                print ' map ahead is ',Map[ijp[0],ijp[1]] , Map[ijp_right[0],ijp_right[1]]
                sys.exit(' in streamfunction_init ')

            """
            -----------------------------------------------------------------------
             go forward in direction
            -----------------------------------------------------------------------
            """
            pyom.line_dir[isle,n,:] = Dir
            ij += Dir
            if pyom.boundary[isle,0,0] == ij[0] and pyom.boundary[isle,0,1] == ij[1]:
                cont = False

            """
            -----------------------------------------------------------------------
             account for cyclic boundary conditions
            -----------------------------------------------------------------------
            """
            if pyom.enable_cyclic_x and Dir[0] == 1 and Dir[1] == 0 and ij[0] > pyom.nx+1:
                if verbose:
                    print ' shifting to western cyclic boundary'
                ij[0] -= pyom.nx
            if pyom.enable_cyclic_x and Dir[0] == -1 and Dir[1] == 0 and ij[0] < 2:
                if verbose:
                    print ' shifting to eastern cyclic boundary'
                ij[0] += pyom.nx
            if pyom.boundary[isle,0,0] == ij[0] and pyom.boundary[isle,0,1] == ij[1]:
                cont = False

            if cont:
                n = n+1
                if n > max_boundary:
                    raise RuntimeError('increase value of max_boundary')
                pyom.boundary[isle,n,:] = ij

        pyom.nr_boundary[isle] = n+1
        print ' number of points is ',n+1
        if verbose:
            print ' '
            print ' Positions:'
            for n in xrange(pyom.nr_boundary[isle]): #n=1,nr_boundary(isle)
                print ' pos=',pyom.boundary[isle,n,:],' dir=',pyom.line_dir[isle,n,:]

    """
    -----------------------------------------------------------------------
     precalculate time independent boundary components of streamfunction
    -----------------------------------------------------------------------
    """
    forc[...] = 0.0
    for isle in xrange(pyom.nisle): #isle=1,nisle
        pyom.psin[:,:,isle] = 0.0
        for n in xrange(pyom.nr_boundary[isle]): #n=1,nr_boundary(isle)
            i = pyom.boundary[isle,n,0]
            j = pyom.boundary[isle,n,1]
            if i >= 0 and i <= pyom.nx+3 and j >= 0 and j <= pyom.ny+3:
                pyom.psin[i,j,isle] = 1.0

        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.psin[:,:,isle])
        print ' solving for boundary contribution by island ',isle

        converged = congrad_streamfunction(forc,pyom.psin[:,:,isle],pyom)
        print ' itts =  ',pyom.congr_itts

        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.psin[:,:,isle])

    """
    -----------------------------------------------------------------------
     precalculate time independent island integrals
    -----------------------------------------------------------------------
    """
    for n in xrange(pyom.nisle): #n=1,nisle
        for isle in xrange(pyom.nisle): #isle=1,nisle
            fpx[...] = 0
            fpy[...] = 0
            fpx[1:, 1:] = -pyom.maskU[1:, 1:, pyom.nz-1]\
                    *(pyom.psin[1:, 1:,isle]-pyom.psin[1:, :pyom.ny+3,isle])\
                    /pyom.dyt[1:]*pyom.hur[1:, 1:]
            fpy[1:, 1:] = pyom.maskV[1:, 1:, pyom.nz-1]\
                    *(pyom.psin[1:, 1:,isle]-pyom.psin[:pyom.nx+3, 1:,isle])\
                    /(pyom.cosu[1:]*pyom.dxt[1:, np.newaxis])*pyom.hvr[1:, 1:]
            #for j in xrange(1, pyom.ny+4): #j=js_pe-onx+1,je_pe+onx
            #    for i in xrange(1, pyom.nx+4): #i=is_pe-onx+1,ie_pe+onx
            #        fpx[i,j] =-pyom.maskU[i,j,pyom.nz-1]*( pyom.psin[i,j,isle]-pyom.psin[i,j-1,isle])/pyom.dyt[j]*pyom.hur[i,j]
            #        fpy[i,j] = pyom.maskV[i,j,pyom.nz-1]*( pyom.psin[i,j,isle]-pyom.psin[i-1,j,isle])/(pyom.cosu[j]*pyom.dxt[i])*pyom.hvr[i,j]
            pyom.line_psin[n,isle] = line_integral(n,fpx,fpy, pyom)

def avoid_cyclic_boundaries(Map, isle, n, pyom):
    for i in xrange(pyom.nx/2+2, pyom.nx+2): #i=nx/2+1,nx
        for j in xrange(1, pyom.ny+2): #j=0,ny
            if Map[i,j] == 1 and Map[i,j+1] == -1:
                #initial direction is eastward, we come from the west
                ij=np.array([i,j])
                cont = True
                Dir = np.array([1,0])
                pyom.boundary[isle,n,:] = np.array([ij[0]-1,ij[1]])
                return (cont, ij, Dir)
            if Map[i,j] == -1 and Map[i,j+1] == 1:
                # initial direction is westward, we come from the east
                ij = np.array([i-1,j])
                cont = True
                Dir = np.array([-1,0])
                pyom.boundary[isle,n,:] = np.array([ij[0]+1,ij[1]])
                return (cont, ij, Dir)
    return (False, None, np.array([0,0]))

def avoid_cyclic_boundaries2(Map, isle, n, pyom):
    for i in xrange(pyom.nx/2,0,-1): #i=nx/2,1,-1  ! avoid starting close to cyclic bondaries
        for j in xrange(1, pyom.ny+2): #j=0,ny
            if Map[i,j] == 1 and Map[i,j+1] == -1:
                # initial direction is eastward, we come from the west
                ij=np.array([i,j])
                cont = True
                Dir = np.array([1,0])
                pyom.boundary[isle,n,:]= np.array([ij[0]-1,ij[1]])
                return (cont, ij, Dir)
            if Map[i,j] == -1 and Map[i,j+1] == 1:
                # initial direction is westward, we come from the east
                ij=np.array([i-1,j])
                cont = True
                Dir = np.array([-1,0])
                pyom.boundary[isle,n,:] = np.array([ij[0]+1,ij[1]])
                return (cont, ij, Dir)
    return (False, None, np.array([0,0]))

def line_integral(isle,uloc,vloc,pyom):
    """
    =======================================================================
     calculate line integral along island isle
    =======================================================================
    """
    #integer :: isle
    #integer :: js_,je_,is_,ie_
    #!real*8 :: uloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #!real*8 :: vloc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: uloc(is_:ie_,js_:je_)
    #real*8 :: vloc(is_:ie_,js_:je_)
    #real*8 :: line
    #integer :: n,i,j,nm1,js,je,is,ie
    line = 0

    for n in xrange(pyom.nr_boundary[isle]): #n=1,nr_boundary(isle)
        nm1 = n-1
        if nm1 < 1:
            nm1 = pyom.nr_boundary[isle]
        i = pyom.boundary[isle,n,0]
        j = pyom.boundary[isle,n,1]
        if i >= 1 and i <= pyom.nx+2 and j >= 1 and j <= pyom.ny+2:
            if pyom.line_dir[isle,n,0] == 1 and pyom.line_dir[isle,n,1] == 0:   # to east
                line += vloc[i,j]*pyom.dyu[j] + uloc[i,j+1]*pyom.dxu[i]*pyom.cost[j+1]
            elif pyom.line_dir[isle,n,0] == -1 and pyom.line_dir[isle,n,1] == 0: # to west
                line -= vloc[i+1,j]*pyom.dyu[j] - uloc[i,j]*pyom.dxu[i]*pyom.cost[j]
            elif pyom.line_dir[isle,n,0] == 0 and pyom.line_dir[isle,n,1] == 1: # to north
                line += vloc[i,j]*pyom.dyu[j]  - uloc[i,j]*pyom.dxu[i]*pyom.cost[j]
            elif pyom.line_dir[isle,n,0] ==  0 and pyom.line_dir[isle,n,1] == -1: # to south
                line += uloc[i,j+1]*pyom.dxu[i]*pyom.cost[j+1] - vloc[i+1,j]*pyom.dyu[j]
            else:
                print ' line_dir =',pyom.line_dir[isle,n,:],' at pos. ',pyom.boundary[isle,n,:]
                sys.exit(' missing line_dir in line integral')
    return line

def mod10(m):
    if m > 0:
        return m % 10
    else:
        return m

def showmap(Map, pyom):
    #integer :: js_,je_,is_,ie_
    #!integer :: map(1-onx:nx+onx,1-onx:ny+onx)
    #integer :: map(is_:ie_,js_:je_)
    #integer,parameter :: linewidth=125
    #integer :: istart,iremain,isweep,iline,i,j,mod10
    #integer :: imt
    linewidth = 125

    imt = pyom.nx + 4
    iremain = imt
    istart = 0
    print("")
    print(" "*(5+min(linewidth,imt)/2-13) + "Land mass and perimeter")
    for isweep in xrange(1, imt/linewidth + 2): #isweep=1,imt/linewidth + 1
        iline = min(iremain, linewidth)
        iremain = iremain - iline
        if iline > 0:
            print ' '
            print("".join(["{:5d}".format(istart+i+1-2) for i in xrange(1,iline+1,5)]))
            for j in xrange(pyom.ny+3, -1, -1): #j=ny+onx,1-onx,-1
                print("{:3d} ".format(j) + "".join([str(int(mod10(Map[istart+i -2,j]))) if mod10(Map[istart+i -2,j]) >= 0 else "*" for i in xrange(2, iline+2)]))
            print("".join(["{:5d}".format(istart+i+1-2) for i in xrange(1,iline+1,5)]))
            #print '(t6,32i5)', (istart+i+4-onx,i=1,iline,5)
            istart = istart + iline
    print("")

def solve_streamfunction(pyom,benchtest=False):
    """
    =======================================================================
      solve for barotropic streamfunction
    =======================================================================
    """
    # use main_module
    # implicit none
    # integer :: i,j,k,isle
    # real*8 :: fxa,line_forc(nisle),line_psi0(nisle),aloc(nisle,nisle)
    # real*8 :: fpx(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    # real*8 :: fpy(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    # real*8 :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    # integer :: ipiv(nisle),info
    # logical :: converged
    line_forc = np.empty(pyom.nisle)
    line_psi0 = np.empty(pyom.nisle)
    aloc      = np.empty((pyom.nisle, pyom.nisle))

    #hydrostatic pressure
    fxa = pyom.grav/pyom.rho_0
    pyom.p_hydro[:,:,pyom.nz-1] = 0.5*pyom.rho[:,:,pyom.nz-1,pyom.tau]*fxa*pyom.dzw[pyom.nz-1]*pyom.maskT[:,:,pyom.nz-1]
    for k in xrange(pyom.nz-2, -1, -1): #k=nz-1,1,-1
        pyom.p_hydro[:,:,k] = pyom.maskT[:,:,k]*(pyom.p_hydro[:,:,k+1]+ 0.5*(pyom.rho[:,:,k+1,pyom.tau]+pyom.rho[:,:,k,pyom.tau])*fxa*pyom.dzw[k])

    # add hydrostatic pressure gradient
    pyom.du[2:pyom.nx+2,2:pyom.ny+2,:,pyom.tau] += \
            (pyom.p_hydro[3:pyom.nx+3,2:pyom.ny+2,:]-pyom.p_hydro[2:pyom.nx+2,2:pyom.ny+2,:]) \
            /(np.ones(pyom.nz)*pyom.cost[2:pyom.ny+2,np.newaxis]*pyom.dxu[2:pyom.nx+2,np.newaxis,np.newaxis]) \
            *pyom.maskU[2:pyom.nx+2,2:pyom.ny+2,:]
    pyom.dv[2:pyom.nx+2,2:pyom.ny+2,:,pyom.tau] += \
            (pyom.p_hydro[2:pyom.nx+2,3:pyom.ny+3,:]-pyom.p_hydro[2:pyom.nx+2,2:pyom.ny+2,:]) \
            /(np.ones(pyom.nz) * pyom.dyu[2:pyom.ny+2, np.newaxis] * np.ones(pyom.nx)[:, np.newaxis, np.newaxis]) \
            *pyom.maskV[2:pyom.nx+2,2:pyom.ny+2,:]
    #for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
    #    for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #        pyom.du[i,j,:,pyom.tau] += -( pyom.p_hydro[i+1,j,:]-pyom.p_hydro[i,j,:]  )/(pyom.dxu[i]*pyom.cost[j]) *pyom.maskU[i,j,:]
    #        pyom.dv[i,j,:,pyom.tau] += -( pyom.p_hydro[i,j+1,:]-pyom.p_hydro[i,j,:]  ) /pyom.dyu[j]*pyom.maskV[i,j,:]

    # forcing for barotropic streamfunction
    fpx = np.zeros((pyom.nx+4, pyom.ny+4))
    fpy = np.zeros((pyom.nx+4, pyom.ny+4))
    fpx += np.add.reduce((pyom.du[:,:,:,pyom.tau]+pyom.du_mix)*pyom.maskU*pyom.dzt, axis=(2,))
    fpy += np.add.reduce((pyom.dv[:,:,:,pyom.tau]+pyom.dv_mix)*pyom.maskV*pyom.dzt, axis=(2,))
    #for k in xrange(pyom.nz): #k=1,nz
    #    fpx += (pyom.du[:,:,k,pyom.tau]+pyom.du_mix[:,:,k])*pyom.maskU[:,:,k]*pyom.dzt[k]
    #    fpy += (pyom.dv[:,:,k,pyom.tau]+pyom.dv_mix[:,:,k])*pyom.maskV[:,:,k]*pyom.dzt[k]

    fpx *= pyom.hur
    fpy *= pyom.hvr

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(fpx)
        cyclic.setcyclic_x(fpy)

    forc = np.empty((pyom.nx+4, pyom.ny+4))
    forc[2:pyom.nx+2, 2:pyom.ny+2] = (fpy[3:pyom.nx+3, 2:pyom.ny+2]-fpy[2:pyom.nx+2, 2:pyom.ny+2]) \
            /(pyom.cosu[2:pyom.ny+2]*pyom.dxu[2:pyom.nx+2, np.newaxis]) \
            -(pyom.cost[3:pyom.ny+3]*fpx[2:pyom.nx+2, 3:pyom.ny+3]-pyom.cost[2:pyom.ny+2]*fpx[2:pyom.nx+2, 2:pyom.ny+2]) \
            /(pyom.cosu[2:pyom.ny+2]*pyom.dyu[2:pyom.ny+2])
    #for j in (2, pyom.ny+2): #j=js_pe,je_pe
    #    for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #        forc[i,j] = (fpy[i+1,j]-fpy[i,j])/(pyom.cosu[j]*pyom.dxu[i])-(pyom.cost[j+1]*fpx[i,j+1]-pyom.cost[j]*fpx[i,j])/(pyom.cosu[j]*pyom.dyu[j])

    # solve for interior streamfunction
    pyom.dpsi[:,:,pyom.taup1] = 2*pyom.dpsi[:,:,pyom.tau]-pyom.dpsi[:,:,pyom.taum1] # first guess, we need three time levels here
    congrad_streamfunction(forc,pyom.dpsi[:,:,pyom.taup1], pyom,benchtest)

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(pyom.dpsi[:,:,pyom.taup1])

    if pyom.nisle > 1:
        # calculate island integrals of forcing, keep psi constant on island 1
        for k in xrange(1,pyom.nisle): #k=2,nisle
            line_forc[k] = line_integral(k,fpx,fpy,pyom)

        # calculate island integrals of interior streamfunction
        for k in xrange(1, pyom.nisle): #k=2,nisle
            fpx[...] = 0.0
            fpy[...] = 0.0
            fpx[1:, 1:] = \
                    -pyom.maskU[1:, 1:, pyom.nz-1] \
                    * (pyom.dpsi[1:, 1:,pyom.taup1]-pyom.dpsi[1:, :pyom.ny+3,pyom.taup1]) \
                    /pyom.dyt[1:] *pyom.hur[1:, 1:]
            fpy[1:, 1:] = \
                    pyom.maskV[1:, 1:, pyom.nz-1] \
                    * (pyom.dpsi[1:, 1:,pyom.taup1]-pyom.dpsi[:pyom.nx+3, 1:,pyom.taup1]) \
                    /(pyom.cosu[1:]*pyom.dxt[1:, np.newaxis])*pyom.hvr[1:,1:]
            #for i in xrange(1, pyom.nx+4): #i=is_pe-onx+1,ie_pe+onx
            #    for j in xrange(1, pyom.ny+4): #j=js_pe-onx+1,je_pe+onx
            #        fpx[i,j] =-pyom.maskU[i,j,pyom.nz-1]*( pyom.dpsi[i,j,pyom.taup1]-pyom.dpsi[i,j-1,pyom.taup1])/pyom.dyt[j]*pyom.hur[i,j]
            #        fpy[i,j] = pyom.maskV[i,j,pyom.nz-1]*( pyom.dpsi[i,j,pyom.taup1]-pyom.dpsi[i-1,j,pyom.taup1])/(pyom.cosu[j]*pyom.dxt[i])*pyom.hvr[i,j]
            line_psi0[k] = line_integral(k,fpx,fpy,pyom)

        line_forc -= line_psi0

        # solve for time dependent boundary values
        aloc[...] = pyom.line_psin # will be changed in lapack routine
        #CALL DGESV(nisle-1 , 1, aloc(2:nisle,2:nisle), nisle-1, IPIV, line_forc(2:nisle), nisle-1, INFO )
        (lu, ipiv, line_forc[1:pyom.nisle], info) = lapack.dgesv(aloc[1:pyom.nisle, 1:pyom.nisle], line_forc[1:pyom.nisle])

        if info != 0:
            print 'info = ',info
            print ' line_forc=',line_forc[1:pyom.nisle]
            sys.exit(' in solve_streamfunction, lapack info not zero ')
        pyom.dpsin[1:pyom.nisle,pyom.tau] = line_forc[1:pyom.nisle]

    # integrate barotropic and baroclinic velocity forward in time
    pyom.psi[:,:,pyom.taup1] = pyom.psi[:,:,pyom.tau]+ pyom.dt_mom*( (1.5+pyom.AB_eps)*pyom.dpsi[:,:,pyom.taup1] - (0.5+pyom.AB_eps)*pyom.dpsi[:,:,pyom.tau] )
    pyom.psi[:, :, pyom.taup1] += pyom.dt_mom*np.add.reduce(( (1.5+pyom.AB_eps)*pyom.dpsin[1:pyom.nisle,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dpsin[1:pyom.nisle,pyom.taum1])*pyom.psin[:,:,1:pyom.nisle], axis=2)
    #for isle in xrange(1, pyom.nisle): #isle=2,nisle
    #    pyom.psi[:,:,pyom.taup1] += pyom.dt_mom*( (1.5+pyom.AB_eps)*pyom.dpsin[isle,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dpsin[isle,pyom.taum1])*pyom.psin[:,:,isle]
    pyom.u[:,:,:,pyom.taup1]   += pyom.dt_mom*( pyom.du_mix+ (1.5+pyom.AB_eps)*pyom.du[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.du[:,:,:,pyom.taum1] )*pyom.maskU
    pyom.v[:,:,:,pyom.taup1]   += pyom.dt_mom*( pyom.dv_mix+ (1.5+pyom.AB_eps)*pyom.dv[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dv[:,:,:,pyom.taum1] )*pyom.maskV

    # subtract incorrect vertical mean from baroclinic velocity
    fpx[...] = 0.0
    fpy[...] = 0.0
    fpx += np.add.reduce(pyom.u[:,:,:,pyom.taup1]*pyom.maskU*pyom.dzt, axis=(2,))
    fpy += np.add.reduce(pyom.v[:,:,:,pyom.taup1]*pyom.maskV*pyom.dzt, axis=(2,))
    #for k in xrange(pyom.nz): #k=1,nz
    #    fpx += pyom.u[:,:,k,pyom.taup1]*pyom.maskU[:,:,k]*pyom.dzt[k]
    #    fpy += pyom.v[:,:,k,pyom.taup1]*pyom.maskV[:,:,k]*pyom.dzt[k]
    pyom.u[:,:,:,pyom.taup1] += (np.ones(pyom.nz)*-fpx[:, :, np.newaxis])*pyom.maskU*(np.ones(pyom.nz)*pyom.hur[:, :, np.newaxis])
    pyom.v[:,:,:,pyom.taup1] += (np.ones(pyom.nz)*-fpy[:, :, np.newaxis])*pyom.maskV*(np.ones(pyom.nz)*pyom.hvr[:, :, np.newaxis])
    #for k in xrange(pyom.nz): #k=1,nz
    #    pyom.u[:,:,k,pyom.taup1] = pyom.u[:,:,k,pyom.taup1]-fpx*pyom.maskU[:,:,k]*pyom.hur
    #    pyom.v[:,:,k,pyom.taup1] = pyom.v[:,:,k,pyom.taup1]-fpy*pyom.maskV[:,:,k]*pyom.hvr

    # add barotropic mode to baroclinic velocity
    pyom.u[2:pyom.nx+2, 2:pyom.ny+2, :, pyom.taup1] += \
            -pyom.maskU[2:pyom.nx+2,2:pyom.ny+2,:]\
            *(np.ones(pyom.nz)*( pyom.psi[2:pyom.nx+2,2:pyom.ny+2,pyom.taup1]-pyom.psi[2:pyom.nx+2,1:pyom.ny+1,pyom.taup1])[:, :,np.newaxis])\
            /(np.ones(pyom.nz)*pyom.dyt[2:pyom.ny+2,np.newaxis]*np.ones(pyom.nx)[:, np.newaxis, np.newaxis])\
            *(np.ones(pyom.nz)*pyom.hur[2:pyom.nx+2,2:pyom.ny+2][:, :, np.newaxis])
    pyom.v[2:pyom.nx+2, 2:pyom.ny+2, :, pyom.taup1] += \
            pyom.maskV[2:pyom.nx+2,2:pyom.ny+2,:]\
            *(np.ones(pyom.nz)*( pyom.psi[2:pyom.nx+2,2:pyom.ny+2,pyom.taup1]-pyom.psi[1:pyom.nx+1,2:pyom.ny+2,pyom.taup1])[:, :,np.newaxis])\
            /(np.ones(pyom.nz)*pyom.cosu[2:pyom.ny+2,np.newaxis]*pyom.dxt[2:pyom.nx+2, np.newaxis, np.newaxis])\
            *(np.ones(pyom.nz)*pyom.hvr[2:pyom.nx+2,2:pyom.ny+2][:, :, np.newaxis])
    #for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #    for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
    #        pyom.u[i,j,:,pyom.taup1] += -pyom.maskU[i,j,:]*( pyom.psi[i,j,pyom.taup1]-pyom.psi[i,j-1,pyom.taup1])/pyom.dyt[j]*pyom.hur[i,j]
    #        pyom.v[i,j,:,pyom.taup1] += pyom.maskV[i,j,:]*( pyom.psi[i,j,pyom.taup1]-pyom.psi[i-1,j,pyom.taup1])/(pyom.cosu[j]*pyom.dxt[i])*pyom.hvr[i,j]


def make_coeff_streamfunction(cf, pyom):
    """
    =======================================================================
             A * p = forc
             res = A * p
             res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk)
    =======================================================================
    """
    #real*8 :: cf(is_:ie_,js_:je_,3,3)
    #cf = np.zeros(ie_-is_+1, je_-js_+1, 3, 3)
    cf[2:pyom.nx+2, 2:pyom.ny+2, 1, 1] -= pyom.hvr[3:pyom.nx+3, 2:pyom.ny+2]/(np.ones(pyom.ny)*pyom.dxu[2:pyom.nx+2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[3:pyom.nx+3, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:pyom.ny+2].T).T**2
    cf[2:pyom.nx+2, 2:pyom.ny+2, 2, 1] += pyom.hvr[3:pyom.nx+3, 2:pyom.ny+2]/(np.ones(pyom.ny)*pyom.dxu[2:pyom.nx+2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[3:pyom.nx+3, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:pyom.ny+2].T).T**2
    cf[2:pyom.nx+2, 2:pyom.ny+2, 1, 1] -= pyom.hvr[2:pyom.nx+2, 2:pyom.ny+2]/(np.ones(pyom.ny)*pyom.dxu[2:pyom.nx+2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[2:pyom.nx+2, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:pyom.ny+2].T).T**2
    cf[2:pyom.nx+2, 2:pyom.ny+2, 0, 1] += pyom.hvr[2:pyom.nx+2, 2:pyom.ny+2]/(np.ones(pyom.ny)*pyom.dxu[2:pyom.nx+2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[2:pyom.nx+2, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:pyom.ny+2].T).T**2

    cf[2:pyom.nx+2, 2:pyom.ny+2, 1, 1] -= pyom.hur[2:pyom.nx+2, 3:pyom.ny+3]/(pyom.dyu[2:pyom.ny+2])/(pyom.dyt[3:pyom.ny+3])*pyom.cost[3:pyom.ny+3]/(pyom.cosu[2:pyom.ny+2])
    cf[2:pyom.nx+2, 2:pyom.ny+2, 1, 2] += pyom.hur[2:pyom.nx+2, 3:pyom.ny+3]/(pyom.dyu[2:pyom.ny+2])/(pyom.dyt[3:pyom.ny+3])*pyom.cost[3:pyom.ny+3]/(pyom.cosu[2:pyom.ny+2])
    cf[2:pyom.nx+2, 2:pyom.ny+2, 1, 1] -= pyom.hur[2:pyom.nx+2, 2:pyom.ny+2]/(pyom.dyu[2:pyom.ny+2])/(pyom.dyt[2:pyom.ny+2])*pyom.cost[2:pyom.ny+2]/(pyom.cosu[2:pyom.ny+2])
    cf[2:pyom.nx+2, 2:pyom.ny+2, 1, 0] += pyom.hur[2:pyom.nx+2, 2:pyom.ny+2]/(pyom.dyu[2:pyom.ny+2])/(pyom.dyt[2:pyom.ny+2])*pyom.cost[2:pyom.ny+2]/(pyom.cosu[2:pyom.ny+2])

    #for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #    for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
    #        cf[i,j, 1, 1] -= pyom.hvr[i+1,j]/pyom.dxu[i]/pyom.dxt[i+1] /pyom.cosu[j]**2
    #        cf[i,j, 2, 1] += pyom.hvr[i+1,j]/pyom.dxu[i]/pyom.dxt[i+1] /pyom.cosu[j]**2
    #        cf[i,j, 1, 1] -= pyom.hvr[i  ,j]/pyom.dxu[i]/pyom.dxt[i  ] /pyom.cosu[j]**2
    #        cf[i,j, 0, 1] += pyom.hvr[i  ,j]/pyom.dxu[i]/pyom.dxt[i  ] /pyom.cosu[j]**2

    #        cf[i,j, 1, 1] -= pyom.hur[i,j+1]/pyom.dyu[j]/pyom.dyt[j+1]*pyom.cost[j+1]/pyom.cosu[j]
    #        cf[i,j, 1, 2] += pyom.hur[i,j+1]/pyom.dyu[j]/pyom.dyt[j+1]*pyom.cost[j+1]/pyom.cosu[j]
    #        cf[i,j, 1, 1] -= pyom.hur[i,j  ]/pyom.dyu[j]/pyom.dyt[j  ]*pyom.cost[j  ]/pyom.cosu[j]
    #        cf[i,j, 1, 0] += pyom.hur[i,j  ]/pyom.dyu[j]/pyom.dyt[j  ]*pyom.cost[j  ]/pyom.cosu[j]


def congrad_streamfunction(forc,sol,pyom,benchtest=False):
    """
    =======================================================================
      conjugate gradient solver with preconditioner from MOM
    =======================================================================
    """
    #use main_module
    #implicit none
    #integer :: is_,ie_,js_,je_
    #integer :: iterations, n,i,j
    #!real*8  :: forc(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #!real*8  :: sol(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: forc(is_:ie_,js_:je_)
    #real*8  :: sol(is_:ie_,js_:je_)
    #real*8  :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: Z(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: Zres(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: ss(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: As(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8  :: estimated_error
    #real*8  :: zresmax,betakm1,betak,betak_min=0,betaquot,s_dot_As,smax
    #real*8  :: alpha,step,step1=0,convergence_rate
    #real*8 , external :: absmax_sfc,dot_sfc
    #logical, save :: first = .true.
    #real*8 , allocatable,save :: cf(:,:,:,:)
    #logical :: converged
    betak_min = 0.0

    # congrad_streamfunction.first is basically like a static variable
    if congrad_streamfunction.first:
        congrad_streamfunction.cf = np.zeros((pyom.nx+4, pyom.ny+4, 3, 3))
        make_coeff_streamfunction(congrad_streamfunction.cf,pyom)
        congrad_streamfunction.first = False

    Z    = np.zeros((pyom.nx+4, pyom.ny+4))
    Zres = np.zeros((pyom.nx+4, pyom.ny+4))
    ss   = np.zeros((pyom.nx+4, pyom.ny+4))
    As   = np.zeros((pyom.nx+4, pyom.ny+4))
    res  = np.zeros((pyom.nx+4, pyom.ny+4))
    """
    -----------------------------------------------------------------------
         make approximate inverse operator Z (always even symmetry)
    -----------------------------------------------------------------------
    """
    make_inv_sfc(congrad_streamfunction.cf, Z, pyom)
    """
    -----------------------------------------------------------------------
         impose boundary conditions on guess
         sol(0) = guess
    -----------------------------------------------------------------------
    """
    #for i in xrange(pyom.nx+4):
    #    for j in xrange(pyom.ny+4):
    #        print "sol", i, j, sol[i,j]
    #sys.exit()
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(sol)
    """
    -----------------------------------------------------------------------
         res(0)  = forc - A * eta(0)
    -----------------------------------------------------------------------
    """
    solve_pressure.apply_op(congrad_streamfunction.cf, sol, pyom, res)
    res[2:pyom.nx+2, 2:pyom.ny+2] = forc[2:pyom.nx+2, 2:pyom.ny+2] - res[2:pyom.nx+2, 2:pyom.ny+2]

    """
    -----------------------------------------------------------------------
         Zres(k-1) = Z * res(k-1)
         see if guess is a solution, bail out to avoid division by zero
    -----------------------------------------------------------------------
    """
    n = 0
    inv_op_sfc(Z, res, Zres, pyom)
    Zresmax = absmax_sfc(Zres, pyom)
    # Assume convergence rate of 0.99 to extrapolate error
    if (100.0 * Zresmax < pyom.congr_epsilon or benchtest):
        estimated_error = 100.0 * Zresmax
        print_info(n, estimated_error, pyom)
        return True #Converged
    """
    -----------------------------------------------------------------------
         beta(0) = 1
         ss(0)    = zerovector()
    -----------------------------------------------------------------------
    """
    betakm1 = 1.0
    ss[...] = 0.
    """
    -----------------------------------------------------------------------
         begin iteration loop
    ----------------------------------------------------------------------
    """
    for n in xrange(1, pyom.congr_max_iterations): #n = 1,congr_max_iterations
        """
        -----------------------------------------------------------------------
               Zres(k-1) = Z * res(k-1)
        -----------------------------------------------------------------------
        """
        inv_op_sfc(Z, res, Zres, pyom)
        """
        -----------------------------------------------------------------------
               beta(k)   = res(k-1) * Zres(k-1)
        -----------------------------------------------------------------------
        """
        betak = dot_sfc(Zres, res, pyom)
        if n == 1:
            betak_min = abs(betak)
        elif n > 2:
            betak_min = min(betak_min, abs(betak))
            if abs(betak) > 100.0*betak_min:
                print 'WARNING: solver diverging at itt=',pyom.congr_itts
                fail(n, estimated_error, pyom)
                return False #Converged
        """
        -----------------------------------------------------------------------
               ss(k)      = Zres(k-1) + (beta(k)/beta(k-1)) * ss(k-1)
        -----------------------------------------------------------------------
        """
        betaquot = betak/betakm1
        ss[2:pyom.nx+2,2:pyom.ny+2] = Zres[2:pyom.nx+2,2:pyom.ny+2] + betaquot*ss[2:pyom.nx+2,2:pyom.ny+2]
        #for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
        #    for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
        #        ss[i,j] = Zres[i,j] + betaquot*ss[i,j]

        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(ss)
        """
        -----------------------------------------------------------------------
               As(k)     = A * ss(k)
        -----------------------------------------------------------------------
        """
        solve_pressure.apply_op(congrad_streamfunction.cf, ss, pyom, As)
        #print "AS", As
        #sys.exit()
        """
        -----------------------------------------------------------------------
               If ss=0 then the division for alpha(k) gives a float exception.
               Assume convergence rate of 0.99 to extrapolate error.
               Also assume alpha(k) ~ 1.
        -----------------------------------------------------------------------
        """
        s_dot_As = dot_sfc(ss, As, pyom)
        if abs(s_dot_As) < abs(betak)*1.e-10:
            smax = absmax_sfc(ss,pyom)
            estimated_error = 100.0 * smax
            print_info(n, estimated_error, pyom)
            return True #Converged
        """
        -----------------------------------------------------------------------
               alpha(k)  = beta(k) / (ss(k) * As(k))
        -----------------------------------------------------------------------
        """
        alpha = betak / s_dot_As
        """
        -----------------------------------------------------------------------
               update values:
               eta(k)   = eta(k-1) + alpha(k) * ss(k)
               res(k)    = res(k-1) - alpha(k) * As(k)
        -----------------------------------------------------------------------
        """
        sol[2:pyom.nx+2, 2:pyom.ny+2] += alpha * ss[2:pyom.nx+2, 2:pyom.ny+2]
        res[2:pyom.nx+2, 2:pyom.ny+2] += -alpha * As[2:pyom.nx+2, 2:pyom.ny+2]
        #for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
        #    for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
        #        sol[i,j] += alpha * ss[i,j]
        #        res[i,j] += -alpha * As[i,j]

        smax = absmax_sfc(ss, pyom)
        """
        -----------------------------------------------------------------------
               test for convergence
               if (estimated_error) < congr_epsilon) exit
        -----------------------------------------------------------------------
        """
        step = abs(alpha) * smax
        if n == 1:
            step1 = step
            estimated_error = step
            if step < pyom.congr_epsilon:
                print_info(n, estimated_error, pyom)
                return True #Converged
        elif step < pyom.congr_epsilon:
            convergence_rate = np.exp(np.log(step/step1)/(n-1))
            estimated_error = step*convergence_rate/(1.0-convergence_rate)
            if estimated_error < pyom.congr_epsilon:
                print_info(n, estimated_error, pyom)
                return True #Converged
        betakm1 = betak
    """
    -----------------------------------------------------------------------
         end of iteration loop
    -----------------------------------------------------------------------
    """
    print ' WARNING: max iterations exceeded at itt=',n
    fail(n, estimated_error, pyom)
    return False #Converged
congrad_streamfunction.first = True

def print_info(n, estimated_error, pyom):
    pyom.congr_itts = n
    if pyom.enable_congrad_verbose:
        print ' estimated error=',estimated_error,'/',pyom.congr_epsilon
        print ' iterations=',n

def fail(n, estimated_error, pyom):
    pyom.congr_itts = n
    print ' estimated error=',estimated_error,'/',pyom.congr_epsilon
    print ' iterations=',n
    # check for NaN
    if np.isnan(estimated_error):
        print ' error is NaN, stopping integration '
        #TODO: Snapshot data
        #call panic_snap
        sys.exit(' in solve_streamfunction')

def absmax_sfc(p1, pyom):
    #use main_module
    #implicit none
    #integer :: is_,ie_,js_,je_
    #!real*8 :: s2,p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: s2,p1(is_:ie_,js_:je_)
    #integer :: i,j
    return np.max(np.abs(p1))
    #s2 = 0
    #for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
    #    for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #        s2 = max( abs(p1[i,j]), s2 )
    #return s2

def dot_sfc(p1, p2, pyom):
    #!real*8 :: s2,p1(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx),p2(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: s2,p1(is_:ie_,js_:je_),p2(is_:ie_,js_:je_)
    #integer :: i,j
    return np.sum(p1[2:pyom.nx+2, 2:pyom.ny+2]*p2[2:pyom.nx+2, 2:pyom.ny+2])
    #s2 = 0.0
    #for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
    #    for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #        s2 += p1[i,j]*p2[i,j]
    #return s2

def inv_op_sfc(Z,res,Zres,pyom):
    """
    -----------------------------------------------------------------------
         apply approximate inverse Z of the operator A
    -----------------------------------------------------------------------
    """
    #integer :: is_,ie_,js_,je_
    #!real*8 :: Z(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #!real*8 :: res(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #!real*8 :: Zres(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8, dimension(is_:ie_,js_:je_) :: Z,res,Zres
    #integer :: i,j
    Zres[2:pyom.nx+2, 2:pyom.ny+2] = Z[2:pyom.nx+2, 2:pyom.ny+2] * res[2:pyom.nx+2, 2:pyom.ny+2]
    #for i in xrange(2, pyom.nx+2): #is_pe, ie_pe
    #    for j in xrange(2, pyom.ny+2): #js_pe, je_pe
    #        Zres[i,j] = Z[i,j] * res[i,j]

def make_inv_sfc(cf,Z,pyom):
    """
    -----------------------------------------------------------------------
         construct an approximate inverse Z to A
    -----------------------------------------------------------------------
    """
    #integer :: is_,ie_,js_,je_
    #!real*8 :: cf(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx,3,3)
    #!real*8 ::  Z(is_pe-onx:ie_pe+onx,js_pe-onx:je_pe+onx)
    #real*8 :: cf(is_:ie_,js_:je_,3,3)
    #real*8 :: Z (is_:ie_,js_:je_)
    #integer :: i,j,isle,n
#
#   copy diagonal coefficients of A to Z
#
    Z[...] = 0
    Z[2:pyom.nx+2, 2:pyom.ny+2] = cf[2:pyom.nx+2, 2:pyom.ny+2,1,1]
    #for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
    #    for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #        Z[i,j] = cf[i,j,1,1]

#
#   now invert Z
#
    Y = Z[2:pyom.nx+2, 2:pyom.ny+2]
    if climate.is_bohrium:
        Y = Y.copy2numpy()
    Y[Y != 0] = 1./Y[Y != 0]
    if climate.is_bohrium:
        Z[2:pyom.nx+2, 2:pyom.ny+2] = Y
    #for j in xrange(2, pyom.ny+2): #j=js_pe,je_pe
    #    for i in xrange(2, pyom.nx+2): #i=is_pe,ie_pe
    #        if Z[i,j] != 0.0:
    #            Z[i,j] = 1./Z[i,j]
    #        # Seems a bit redundant
    #        #else:
    #        #  Z(i,j) = 0.0

#
#   make inverse zero on island perimeters that are not integrated
#
    if climate.is_bohrium:
        boundary = pyom.boundary.copy2numpy()
    else:
        boundary = pyom.boundary
    for isle in xrange(pyom.nisle): #isle=1,nisle
        for n in xrange(pyom.nr_boundary[isle]): #n=1,nr_boundary(isle)
            i = boundary[isle,n,0]
            j = boundary[isle,n,1]
            if i >= 0 and i <= pyom.nx+3 and j >= 0 and j <= pyom.ny+3:
                Z[i,j] = 0.0
