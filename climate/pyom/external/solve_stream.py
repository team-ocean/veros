"""
solve two dimensional Possion equation
     A * dpsi = forc,  where A = nabla_h^2
with Dirichlet boundary conditions
used for streamfunction
"""

import numpy as np

import climate
from climate.pyom.external import solve_pressure, island
from climate.pyom import cyclic, diagnostics


def streamfunction_init(pyom):
    """
    prepare for island integrals
    """
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
        Map = Map.copy2numpy()
        allmap = allmap.copy2numpy()
        iperm = iperm.copy2numpy()
        jperm = jperm.copy2numpy()
        nippts = nippts.copy2numpy()
        iofs = iofs.copy2numpy()

    print("Initializing streamfunction method")
    verbose = pyom.enable_congrad_verbose
    """
    communicate kbot to get the entire land map
    """
    kmt = np.zeros((pyom.nx+4, pyom.ny+4)) # note that routine will modify kmt
    kmt[2:-2, 2:-2] = (pyom.kbot[2:-2, 2:-2] > 0) * 5

    if pyom.enable_cyclic_x:
        kmt[-2:] = kmt[2:4]
        kmt[0:2] = kmt[-4:-2]

    """
    preprocess land map using MOMs algorithm for B-grid to determine number of islands
    """
    print(" starting MOMs algorithm for B-grid to determine number of islands")
    island.isleperim(kmt,allmap, iperm, jperm, iofs, nippts, pyom.nx+4, pyom.ny+4, mnisle, maxipp,pyom,change_nisle=True, verbose=True)
    if pyom.enable_cyclic_x:
        allmap[-2:] = allmap[2:4]
        allmap[0:2] = allmap[-4:-2]
    _showmap(allmap, pyom)

    """
    allocate variables
    """
    pyom.boundary_mask = np.zeros((pyom.nisle, pyom.nx+4, pyom.ny+4)).astype(np.bool)
    pyom.line_dir_south_mask = np.zeros((pyom.nisle, pyom.nx+4, pyom.ny+4)).astype(np.bool)
    pyom.line_dir_north_mask = np.zeros((pyom.nisle, pyom.nx+4, pyom.ny+4)).astype(np.bool)
    pyom.line_dir_east_mask = np.zeros((pyom.nisle, pyom.nx+4, pyom.ny+4)).astype(np.bool)
    pyom.line_dir_west_mask = np.zeros((pyom.nisle, pyom.nx+4, pyom.ny+4)).astype(np.bool)
    pyom.psin = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nisle))
    pyom.dpsin = np.zeros((pyom.nisle, 3))
    pyom.line_psin = np.zeros((pyom.nisle, pyom.nisle))

    if climate.is_bohrium:
        pyom.boundary_mask = pyom.boundary_mask.copy2numpy()
        pyom.line_dir_south_mask = pyom.line_dir_south_mask.copy2numpy()
        pyom.line_dir_north_mask = pyom.line_dir_north_mask.copy2numpy()
        pyom.line_dir_east_mask  = pyom.line_dir_east_mask.copy2numpy()
        pyom.line_dir_west_mask  = pyom.line_dir_west_mask.copy2numpy()

    for isle in xrange(pyom.nisle): #isle=1,nisle
        print(" ------------------------")
        print(" processing island #{:d}".format(isle))
        print(" ------------------------")

        """
        land map for island number isle: 1 is land, -1 is perimeter, 0 is ocean
        """
        kmt[...] = allmap != isle+1
        island.isleperim(kmt,Map, iperm, jperm, iofs, nippts, pyom.nx+4, pyom.ny+4, mnisle, maxipp,pyom)
        if verbose:
            _showmap(Map, pyom)

        """
        find a starting point
        """
        n = 0
        # avoid starting close to cyclic bondaries
        (cont, ij, Dir, startPos) = avoid_cyclic_boundaries(Map, isle, n, xrange(pyom.nx/2+2, pyom.nx+2), pyom)
        if not cont:
            (cont, ij, Dir, startPos) = avoid_cyclic_boundaries(Map, isle, n, xrange(pyom.nx/2,0,-1), pyom)
            if not cont:
                raise RuntimeError("found no starting point for line integral")

        print(" starting point of line integral is {!r}".format(startPos))
        print(" starting direction is {!r}".format(Dir))

        """
        now find connecting lines
        """
        n = 1
        pyom.boundary_mask[isle,ij[0],ij[1]] = 1
        cont = True
        while cont:
            """
            consider map in front of line direction and to the right and decide where to go
            """
            if Dir[0]== 0 and Dir[1]== 1: # north
                ijp      = [ij[0]  ,ij[1]+1]
                ijp_right= [ij[0]+1,ij[1]+1]
            elif Dir[0]==-1 and Dir[1]== 0: # west
                ijp      = [ij[0]  ,ij[1]  ]
                ijp_right= [ij[0]  ,ij[1]+1]
            elif Dir[0]== 0 and Dir[1]==-1: # south
                ijp      = [ij[0]+1,ij[1]  ]
                ijp_right= [ij[0]  ,ij[1]  ]
            elif Dir[0]== 1 and Dir[1]== 0: # east
                ijp      = [ij[0]+1,ij[1]+1]
                ijp_right= [ij[0]+1,ij[1]  ]

            """
            4 cases are possible
            """

            if verbose:
                print(" ")
                print(" position is {!r}".format(ij))
                print(" direction is {!r}".format(Dir))
                print(" map ahead is {} {}".format(Map[ijp[0],ijp[1]], Map[ijp_right[0],ijp_right[1]]))

            if Map[ijp[0],ijp[1]] == -1 and Map[ijp_right[0],ijp_right[1]] == 1:
                if verbose:
                    print(" go forward")
            elif Map[ijp[0],ijp[1]] == -1 and Map[ijp_right[0],ijp_right[1]] == -1:
                if verbose:
                    print(" turn right")
                Dir = [Dir[1],-Dir[0]]
            elif Map[ijp[0],ijp[1]] == 1 and Map[ijp_right[0],ijp_right[1]] == 1:
                if verbose:
                    print(" turn left")
                Dir = [-Dir[1],Dir[0]]
            elif Map[ijp[0],ijp[1]] == 1 and Map[ijp_right[0],ijp_right[1]] == -1:
                if verbose:
                    print(" turn left")
                Dir = [-Dir[1],Dir[0]]
            else:
                print(" map ahead is {} {}".format(Map[ijp[0],ijp[1]], Map[ijp_right[0],ijp_right[1]]))
                raise RuntimeError("unknown situation or lost track")

            """
            go forward in direction
            """
            if Dir[0]== 0 and Dir[1]== 1: #north
                pyom.line_dir_north_mask[isle, ij[0], ij[1]] = 1
            elif Dir[0]==-1 and Dir[1]== 0: #west
                pyom.line_dir_west_mask[isle, ij[0], ij[1]] = 1
            elif Dir[0]== 0 and Dir[1]==-1: #south
                pyom.line_dir_south_mask[isle, ij[0], ij[1]] = 1
            elif Dir[0]== 1 and Dir[1]== 0: #east
                pyom.line_dir_east_mask[isle, ij[0], ij[1]] = 1
            ij = [ij[0] + Dir[0], ij[1] + Dir[1]]
            if startPos[0] == ij[0] and startPos[1] == ij[1]:
                cont = False

            """
            account for cyclic boundary conditions
            """
            if pyom.enable_cyclic_x and Dir[0] == 1 and Dir[1] == 0 and ij[0] > pyom.nx+1:
                if verbose:
                    print(" shifting to western cyclic boundary")
                ij[0] -= pyom.nx
            if pyom.enable_cyclic_x and Dir[0] == -1 and Dir[1] == 0 and ij[0] < 2:
                if verbose:
                    print(" shifting to eastern cyclic boundary")
                ij[0] += pyom.nx
            if startPos[0] == ij[0] and startPos[1] == ij[1]:
                cont = False

            if cont:
                n += 1
                pyom.boundary_mask[isle,ij[0],ij[1]] = 1

        print(" number of points is {:d}".format(n+1))
        if verbose:
            print(" ")
            print(" Positions:")
            print(" boundary: {!r}".format(pyom.boundary_mask[isle]))

    if climate.is_bohrium:
        pyom.boundary_mask = np.array(pyom.boundary_mask)
        pyom.line_dir_south_mask = np.array(pyom.line_dir_south_mask)
        pyom.line_dir_north_mask = np.array(pyom.line_dir_north_mask)
        pyom.line_dir_east_mask  = np.array(pyom.line_dir_east_mask)
        pyom.line_dir_west_mask  = np.array(pyom.line_dir_west_mask)


    """
    precalculate time independent boundary components of streamfunction
    """
    forc[...] = 0.0
    for isle in xrange(pyom.nisle):
        pyom.psin[:,:,isle] = pyom.boundary_mask[isle]

        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.psin[:,:,isle])
        print(" solving for boundary contribution by island {:d}".format(isle))

        congrad_streamfunction(forc,pyom.psin[:,:,isle],pyom)
        print(" itts =  {:d}".format(pyom.congr_itts))

        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(pyom.psin[:,:,isle])

    """
    precalculate time independent island integrals
    """
    for n in xrange(pyom.nisle):
        for isle in xrange(pyom.nisle):
            fpx[...] = 0
            fpy[...] = 0
            fpx[1:, 1:] = -pyom.maskU[1:, 1:, -1] \
                    * (pyom.psin[1:, 1:, isle] - pyom.psin[1:, :-1, isle]) \
                    / pyom.dyt[1:] * pyom.hur[1:, 1:]
            fpy[1:, 1:] = pyom.maskV[1:, 1:, -1] \
                    * (pyom.psin[1:, 1:, isle] - pyom.psin[:-1, 1:, isle]) \
                    / (pyom.cosu[1:] * pyom.dxt[1:, np.newaxis]) * pyom.hvr[1:, 1:]
            pyom.line_psin[n,isle] = line_integral(n,fpx,fpy, pyom)


def avoid_cyclic_boundaries(Map, isle, n, x_range, pyom):
    for i in x_range: #i=nx/2+1,nx
        for j in xrange(1, pyom.ny+2): #j=0,ny
            if Map[i,j] == 1 and Map[i,j+1] == -1:
                #initial direction is eastward, we come from the west
                cont = True
                Dir = [1,0]
                pyom.line_dir_east_mask[isle, i-1, j] = 1
                pyom.boundary_mask[isle, i-1, j] = 1
                return cont, (i,j), Dir, (i-1, j)
            if Map[i,j] == -1 and Map[i,j+1] == 1:
                # initial direction is westward, we come from the east
                cont = True
                Dir = [-1,0]
                pyom.line_dir_west_mask[isle, i, j] = 1
                pyom.boundary_mask[isle, i, j] = 1
                return cont, (i-1,j), Dir, (i,j)
    return False, None, [0,0], [0,0]


def line_integral(isle,uloc,vloc,pyom):
    """
    calculate line integral along island isle
    """
    east = vloc[1:-2,1:-2] * pyom.dyu[np.newaxis, 1:-2] + uloc[1:-2,2:-1] * pyom.dxu[1:-2, np.newaxis] * pyom.cost[np.newaxis,2:-1]
    west = -vloc[2:-1,1:-2] * pyom.dyu[np.newaxis, 1:-2] - uloc[1:-2,1:-2] * (pyom.cost[1:-2]*pyom.dxu[1:-2,np.newaxis])
    north = vloc[1:-2,1:-2] * pyom.dyu[np.newaxis,1:-2]  - uloc[1:-2,1:-2] * (pyom.cost[1:-2]*pyom.dxu[1:-2,np.newaxis])
    south = -vloc[2:-1,1:-2] * pyom.dyu[np.newaxis, 1:-2] + uloc[1:-2,2:-1] * (pyom.cost[2:-1]*pyom.dxu[1:-2, np.newaxis])
    east = np.sum(east * (pyom.line_dir_east_mask[isle,1:-2,1:-2] & pyom.boundary_mask[isle,1:-2,1:-2]))
    west = np.sum(west * (pyom.line_dir_west_mask[isle,1:-2,1:-2] & pyom.boundary_mask[isle,1:-2,1:-2]))
    north = np.sum(north * (pyom.line_dir_north_mask[isle,1:-2,1:-2] & pyom.boundary_mask[isle,1:-2,1:-2]))
    south = np.sum(south * (pyom.line_dir_south_mask[isle,1:-2,1:-2] & pyom.boundary_mask[isle,1:-2,1:-2]))
    return east + west + north + south


def _mod10(m):
    if m > 0:
        return m % 10
    else:
        return m


def _showmap(Map, pyom):
    linewidth = 125
    imt = pyom.nx + 4
    iremain = imt
    istart = 0
    print("")
    print(" "*(5+min(linewidth,imt)/2-13) + "Land mass and perimeter")
    for isweep in xrange(1, imt/linewidth + 2):
        iline = min(iremain, linewidth)
        iremain = iremain - iline
        if iline > 0:
            print(" ")
            print("".join(["{:5d}".format(istart+i+1-2) for i in xrange(1,iline+1,5)]))
            for j in xrange(pyom.ny+3, -1, -1):
                print("{:3d} ".format(j) + "".join([str(int(_mod10(Map[istart+i -2,j]))) if _mod10(Map[istart+i -2,j]) >= 0 else "*" for i in xrange(2, iline+2)]))
            print("".join(["{:5d}".format(istart+i+1-2) for i in xrange(1,iline+1,5)]))
            istart = istart + iline
    print("")


def solve_streamfunction(pyom,benchtest=False):
    """
    solve for barotropic streamfunction
    """
    line_forc = np.empty(pyom.nisle)
    line_psi0 = np.empty(pyom.nisle)
    aloc      = np.empty((pyom.nisle, pyom.nisle))

    #hydrostatic pressure
    fxa = pyom.grav/pyom.rho_0
    tmp = 0.5*(pyom.rho[:,:,:,pyom.tau])*fxa*pyom.dzw*pyom.maskT
    pyom.p_hydro[:,:,pyom.nz-1] = tmp[:,:,-1] #0.5*pyom.rho[:,:,pyom.nz-1,pyom.tau]*fxa*pyom.dzw[pyom.nz-1]*pyom.maskT[:,:,pyom.nz-1]
    tmp[:,:,:-1] += 0.5*pyom.rho[:,:,1:,pyom.tau]*fxa*pyom.dzw[:-1]*pyom.maskT[:,:,:-1]
    for k in xrange(pyom.nz-2, -1, -1): #k=nz-1,1,-1
        pyom.p_hydro[:,:,k] = pyom.maskT[:,:,k]*pyom.p_hydro[:,:,k+1]+ tmp[:,:,k] #0.5*(pyom.rho[:,:,k+1,pyom.tau]+pyom.rho[:,:,k,pyom.tau])*fxa*pyom.dzw[k])

    # add hydrostatic pressure gradient
    pyom.du[2:-2,2:-2,:,pyom.tau] += \
            -(pyom.p_hydro[3:-1,2:-2,:]-pyom.p_hydro[2:-2,2:-2,:]) \
            / (pyom.cost[np.newaxis,2:-2,np.newaxis] * pyom.dxu[2:-2,np.newaxis,np.newaxis]) \
            *pyom.maskU[2:-2,2:-2,:]
    pyom.dv[2:-2,2:-2,:,pyom.tau] += \
            -(pyom.p_hydro[2:-2,3:-1,:]-pyom.p_hydro[2:-2,2:-2,:]) \
            / pyom.dyu[np.newaxis, 2:-2, np.newaxis] \
            * pyom.maskV[2:-2,2:-2,:]

    # forcing for barotropic streamfunction
    fpx = np.zeros((pyom.nx+4, pyom.ny+4))
    fpy = np.zeros((pyom.nx+4, pyom.ny+4))
    fpx += np.sum((pyom.du[:,:,:,pyom.tau] + pyom.du_mix) * pyom.maskU * pyom.dzt, axis=(2,))
    fpy += np.sum((pyom.dv[:,:,:,pyom.tau] + pyom.dv_mix) * pyom.maskV * pyom.dzt, axis=(2,))

    fpx *= pyom.hur
    fpy *= pyom.hvr

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(fpx)
        cyclic.setcyclic_x(fpy)

    forc = np.empty((pyom.nx+4, pyom.ny+4))
    forc[2:-2, 2:-2] = (fpy[3:-1, 2:-2] - fpy[2:-2, 2:-2]) \
            / (pyom.cosu[2:-2] * pyom.dxu[2:-2, np.newaxis]) \
            - (pyom.cost[3:-1] * fpx[2:-2, 3:-1] - pyom.cost[2:-2] * fpx[2:-2, 2:-2]) \
            / (pyom.cosu[2:-2] * pyom.dyu[2:-2])

    # solve for interior streamfunction
    pyom.dpsi[:,:,pyom.taup1] = 2 * pyom.dpsi[:,:,pyom.tau] - pyom.dpsi[:,:,pyom.taum1] # first guess, we need three time levels here
    congrad_streamfunction(forc,pyom.dpsi[:,:,pyom.taup1], pyom) #NOTE: This fucks with wall time

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
                    * (pyom.dpsi[1:, 1:,pyom.taup1]-pyom.dpsi[1:, :-1,pyom.taup1]) \
                    /pyom.dyt[1:] *pyom.hur[1:, 1:]
            fpy[1:, 1:] = \
                    pyom.maskV[1:, 1:, pyom.nz-1] \
                    * (pyom.dpsi[1:, 1:,pyom.taup1]-pyom.dpsi[:-1, 1:,pyom.taup1]) \
                    /(pyom.cosu[1:]*pyom.dxt[1:, np.newaxis])*pyom.hvr[1:,1:]
            line_psi0[k] = line_integral(k,fpx,fpy,pyom)

        line_forc -= line_psi0

        # solve for time dependent boundary values
        #aloc[...] = pyom.line_psin # will be changed in lapack routine
        #CALL DGESV(nisle-1 , 1, aloc(2:nisle,2:nisle), nisle-1, IPIV, line_forc(2:nisle), nisle-1, INFO )
        if climate.is_bohrium:
            line_forc[1:pyom.nisle] = np.linalg.jacobi(pyom.line_psin[1:pyom.nisle, 1:pyom.nisle], line_forc[1:pyom.nisle])
        else:
            line_forc[1:pyom.nisle] = np.linalg.solve(pyom.line_psin[1:pyom.nisle, 1:pyom.nisle], line_forc[1:pyom.nisle])
        #(lu, ipiv, line_forc[1:pyom.nisle], info) = lapack.dgesv(aloc[1:pyom.nisle, 1:pyom.nisle], line_forc[1:pyom.nisle])

        #if info != 0:
        #    print("info = "),info
        #    print(" line_forc="),line_forc[1:pyom.nisle]
        #    sys.exit(" in solve_streamfunction, lapack info not zero ")
        pyom.dpsin[1:pyom.nisle,pyom.tau] = line_forc[1:pyom.nisle]

    # integrate barotropic and baroclinic velocity forward in time
    pyom.psi[:,:,pyom.taup1] = pyom.psi[:,:,pyom.tau]+ pyom.dt_mom*( (1.5+pyom.AB_eps)*pyom.dpsi[:,:,pyom.taup1] - (0.5+pyom.AB_eps)*pyom.dpsi[:,:,pyom.tau] )
    pyom.psi[:, :, pyom.taup1] += pyom.dt_mom * np.sum(((1.5+pyom.AB_eps)*pyom.dpsin[1:pyom.nisle,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dpsin[1:pyom.nisle,pyom.taum1]) * pyom.psin[:,:,1:pyom.nisle], axis=2)
    pyom.u[:,:,:,pyom.taup1] = pyom.u[:,:,:,pyom.tau] + pyom.dt_mom*( pyom.du_mix+ (1.5+pyom.AB_eps)*pyom.du[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.du[:,:,:,pyom.taum1] )*pyom.maskU
    pyom.v[:,:,:,pyom.taup1] = pyom.v[:,:,:,pyom.tau] + pyom.dt_mom*( pyom.dv_mix+ (1.5+pyom.AB_eps)*pyom.dv[:,:,:,pyom.tau] - (0.5+pyom.AB_eps)*pyom.dv[:,:,:,pyom.taum1] )*pyom.maskV

    # subtract incorrect vertical mean from baroclinic velocity
    fpx = np.sum(pyom.u[:,:,:,pyom.taup1]*pyom.maskU*pyom.dzt, axis=(2,))
    fpy = np.sum(pyom.v[:,:,:,pyom.taup1]*pyom.maskV*pyom.dzt, axis=(2,))
    pyom.u[:,:,:,pyom.taup1] += (np.ones(pyom.nz)*-fpx[:, :, np.newaxis])*pyom.maskU*(np.ones(pyom.nz)*pyom.hur[:, :, np.newaxis])
    pyom.v[:,:,:,pyom.taup1] += (np.ones(pyom.nz)*-fpy[:, :, np.newaxis])*pyom.maskV*(np.ones(pyom.nz)*pyom.hvr[:, :, np.newaxis])

    # add barotropic mode to baroclinic velocity
    pyom.u[2:-2, 2:-2, :, pyom.taup1] += \
            -pyom.maskU[2:-2,2:-2,:]\
            *(np.ones(pyom.nz) * (pyom.psi[2:-2,2:-2,pyom.taup1]-pyom.psi[2:-2,1:-3,pyom.taup1])[:, :,np.newaxis])\
            /(np.ones(pyom.nz) * pyom.dyt[2:-2,np.newaxis]*np.ones(pyom.nx)[:, np.newaxis, np.newaxis])\
            *(np.ones(pyom.nz) * pyom.hur[2:-2,2:-2][:, :, np.newaxis])
    pyom.v[2:-2, 2:-2, :, pyom.taup1] += \
            pyom.maskV[2:-2,2:-2,:]\
            *(np.ones(pyom.nz) * (pyom.psi[2:-2,2:-2,pyom.taup1]-pyom.psi[1:-3,2:-2,pyom.taup1])[:, :,np.newaxis])\
            /(np.ones(pyom.nz) * pyom.cosu[2:-2,np.newaxis]*pyom.dxt[2:-2, np.newaxis, np.newaxis])\
            *(np.ones(pyom.nz) * pyom.hvr[2:-2,2:-2][:, :, np.newaxis])


def make_coeff_streamfunction(cf, pyom):
    """
    A * p = forc
    res = A * p
    res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk)
    """
    cf[2:-2, 2:-2, 1, 1] -= pyom.hvr[3:-1, 2:-2]/(np.ones(pyom.ny)*pyom.dxu[2:-2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[3:-1, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:-2].T).T**2
    cf[2:-2, 2:-2, 2, 1] += pyom.hvr[3:-1, 2:-2]/(np.ones(pyom.ny)*pyom.dxu[2:-2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[3:-1, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:-2].T).T**2
    cf[2:-2, 2:-2, 1, 1] -= pyom.hvr[2:-2, 2:-2]/(np.ones(pyom.ny)*pyom.dxu[2:-2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[2:-2, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:-2].T).T**2
    cf[2:-2, 2:-2, 0, 1] += pyom.hvr[2:-2, 2:-2]/(np.ones(pyom.ny)*pyom.dxu[2:-2, np.newaxis])/(np.ones(pyom.ny)*pyom.dxt[2:-2, np.newaxis])/(np.ones(pyom.nx)*pyom.cosu[np.newaxis, 2:-2].T).T**2

    cf[2:-2, 2:-2, 1, 1] -= pyom.hur[2:-2, 3:-1]/(pyom.dyu[2:-2])/(pyom.dyt[3:-1])*pyom.cost[3:-1]/(pyom.cosu[2:-2])
    cf[2:-2, 2:-2, 1, 2] += pyom.hur[2:-2, 3:-1]/(pyom.dyu[2:-2])/(pyom.dyt[3:-1])*pyom.cost[3:-1]/(pyom.cosu[2:-2])
    cf[2:-2, 2:-2, 1, 1] -= pyom.hur[2:-2, 2:-2]/(pyom.dyu[2:-2])/(pyom.dyt[2:-2])*pyom.cost[2:-2]/(pyom.cosu[2:-2])
    cf[2:-2, 2:-2, 1, 0] += pyom.hur[2:-2, 2:-2]/(pyom.dyu[2:-2])/(pyom.dyt[2:-2])*pyom.cost[2:-2]/(pyom.cosu[2:-2])


def congrad_streamfunction(forc,sol,pyom):
    """
    conjugate gradient solver with preconditioner from MOM
    """
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
    make approximate inverse operator Z (always even symmetry)
    """
    make_inv_sfc(congrad_streamfunction.cf, Z, pyom)
    """
    impose boundary conditions on guess
    sol(0) = guess
    """
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(sol)
    """
    res(0)  = forc - A * eta(0)
    """
    solve_pressure.apply_op(congrad_streamfunction.cf, sol, res, pyom)
    res[2:-2, 2:-2] = forc[2:-2, 2:-2] - res[2:-2, 2:-2]

    """
    Zres(k-1) = Z * res(k-1)
    see if guess is a solution, bail out to avoid division by zero
    """
    n = np.int(0)
    inv_op_sfc(Z, res, Zres, pyom)
    Zresmax = absmax_sfc(Zres, pyom)
    # Assume convergence rate of 0.99 to extrapolate error
    if 100.0 * Zresmax < pyom.congr_epsilon:
        estimated_error = 100.0 * Zresmax
        print_info(n, estimated_error, pyom)
        return True #Converged
    """
    beta(0) = 1
    ss(0)    = zerovector()
    """
    betakm1 = np.float(1.0)
    ss[...] = 0.
    """
    begin iteration loop
    """
    n = 1
    cont = True
    while n < pyom.congr_max_iterations and cont:
        """
        Zres(k-1) = Z * res(k-1)
        """
        inv_op_sfc(Z, res, Zres, pyom)
        """
        beta(k)   = res(k-1) * Zres(k-1)
        """
        betak = dot_sfc(Zres, res, pyom)
        if n == 1:
            betak_min = np.abs(betak)
        elif n > 2:
            betak_min = np.minimum(betak_min, np.abs(betak))
            if np.abs(betak) > 100.0*betak_min:
                print("WARNING: solver diverging at itt={:d}".format(pyom.congr_itts))
                fail(n, estimated_error, pyom)
                cont = False
                #converged = False #Converged
        """
        ss(k)      = Zres(k-1) + (beta(k)/beta(k-1)) * ss(k-1)
        """
        betaquot = betak/betakm1
        ss[2:-2,2:-2] = Zres[2:-2,2:-2] + betaquot*ss[2:-2,2:-2]

        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(ss)
        """
        As(k)     = A * ss(k)
        """
        solve_pressure.apply_op(congrad_streamfunction.cf, ss, As, pyom)
        """
        If ss=0 then the division for alpha(k) gives a float exception.
        Assume convergence rate of 0.99 to extrapolate error.
        Also assume alpha(k) ~ 1.
        """
        s_dot_As = dot_sfc(ss, As, pyom)
        if np.abs(s_dot_As) < np.abs(betak)*np.float(1.e-10):
            smax = absmax_sfc(ss,pyom)
            estimated_error = 100.0 * smax
            print_info(n, estimated_error, pyom)
            cont = False
            #converged = True #Converged
        """
        alpha(k)  = beta(k) / (ss(k) * As(k))
        """
        alpha = betak / s_dot_As
        """
        update values:
        eta(k)   = eta(k-1) + alpha(k) * ss(k)
        res(k)    = res(k-1) - alpha(k) * As(k)
        """
        if cont:
            sol[2:-2, 2:-2] += alpha * ss[2:-2, 2:-2]
            res[2:-2, 2:-2] += -alpha * As[2:-2, 2:-2]

        smax = absmax_sfc(ss, pyom)
        """
        test for convergence
        if (estimated_error) < congr_epsilon) exit
        """
        step = np.abs(alpha) * smax
        if n == 1:
            step1 = step
            estimated_error = step
            if step < pyom.congr_epsilon:
                print_info(n, estimated_error, pyom)
                cont = False
                #converged = True #Converged
        elif step < pyom.congr_epsilon:
            convergence_rate = np.exp(np.log(step/step1)/(n-1))
            estimated_error = step*convergence_rate/(1.0-convergence_rate)
            if estimated_error < pyom.congr_epsilon:
                print_info(n, estimated_error, pyom)
                cont = False
                #converged = True #Converged
        betakm1 = betak
        if cont:
            n += 1
    """
    end of iteration loop
    """
    if cont:
        print(" WARNING: max iterations exceeded at itt="),n
        fail(n, estimated_error, pyom)
        #return False #Converged
congrad_streamfunction.first = True


def print_info(n, estimated_error, pyom):
    pyom.congr_itts = n
    #if pyom.enable_congrad_verbose:
    #    print(" estimated error="),estimated_error,"/",pyom.congr_epsilon
    #    print(" iterations="),n


def fail(n, estimated_error, pyom):
    pyom.congr_itts = n
    #print(" estimated error="),estimated_error,"/",pyom.congr_epsilon
    #print(" iterations="),n
    # check for NaN
    if np.isnan(estimated_error):
        raise RuntimeError("error is NaN, stopping integration")


def absmax_sfc(p1, pyom):
    return np.max(np.abs(p1))


def dot_sfc(p1, p2, pyom):
    return np.sum(p1[2:-2, 2:-2]*p2[2:-2, 2:-2])


def inv_op_sfc(Z,res,Zres,pyom):
    """
    apply approximate inverse Z of the operator A
    """
    Zres[2:-2, 2:-2] = Z[2:-2, 2:-2] * res[2:-2, 2:-2]


def make_inv_sfc(cf,Z,pyom):
    """
    construct an approximate inverse Z to A
    """
#
#   copy diagonal coefficients of A to Z
#
    Z[...] = 0
    Z[2:-2, 2:-2] = cf[2:-2, 2:-2,1,1]

#
#   now invert Z
#
    Y = Z[2:-2, 2:-2]
    if climate.is_bohrium:
        Y[...] = (1. / (Y+(Y==0)))*(Y!=0)
    else:
        Y[Y != 0] = 1./Y[Y != 0]

#
#   make inverse zero on island perimeters that are not integrated
#
    for isle in xrange(pyom.nisle): #isle=1,nisle
        Z *= np.invert(pyom.boundary_mask[isle]).astype(np.int)
