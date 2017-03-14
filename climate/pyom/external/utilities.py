from climate.pyom import pyom_method

@pyom_method
def line_integrals(pyom,uloc,vloc,kind="same"):
    """
    calculate line integrals along all islands

    :param kind: "same" calculates only line integral contributions of an island with itself,
                 while "full" calculates all possible pairings between all islands
    """
    if kind == "same":
        s1 = s2 = (slice(None),slice(None),slice(None))
    elif kind == "full":
        s1 = (slice(None),slice(None),np.newaxis,slice(None))
        s2 = (slice(None),slice(None),slice(None),np.newaxis)
    else:
        raise ValueError("kind must be 'same' or 'full'")
    east = vloc[1:-2,1:-2,:] * pyom.dyu[np.newaxis, 1:-2, np.newaxis] \
                                + uloc[1:-2,2:-1,:] \
                                    * pyom.dxu[1:-2, np.newaxis, np.newaxis] \
                                    * pyom.cost[np.newaxis,2:-1,np.newaxis]
    west = -vloc[2:-1,1:-2,:] * pyom.dyu[np.newaxis, 1:-2, np.newaxis] \
                                - uloc[1:-2,1:-2,:] \
                                    * pyom.dxu[1:-2, np.newaxis, np.newaxis] \
                                    * pyom.cost[np.newaxis,1:-2,np.newaxis]
    north = vloc[1:-2,1:-2,:] * pyom.dyu[np.newaxis, 1:-2, np.newaxis] \
                                - uloc[1:-2,1:-2,:] \
                                    * pyom.dxu[1:-2, np.newaxis, np.newaxis] \
                                    * pyom.cost[np.newaxis,1:-2,np.newaxis]
    south = -vloc[2:-1,1:-2,:] * pyom.dyu[np.newaxis, 1:-2, np.newaxis] \
                                + uloc[1:-2,2:-1,:] \
                                    * pyom.dxu[1:-2, np.newaxis, np.newaxis] \
                                    * pyom.cost[np.newaxis,2:-1,np.newaxis]
    east = np.sum(east[s1] * (pyom.line_dir_east_mask[1:-2,1:-2] & pyom.boundary_mask[1:-2,1:-2])[s2], axis=(0,1))
    west = np.sum(west[s1] * (pyom.line_dir_west_mask[1:-2,1:-2] & pyom.boundary_mask[1:-2,1:-2])[s2], axis=(0,1))
    north = np.sum(north[s1] * (pyom.line_dir_north_mask[1:-2,1:-2] & pyom.boundary_mask[1:-2,1:-2])[s2], axis=(0,1))
    south = np.sum(south[s1] * (pyom.line_dir_south_mask[1:-2,1:-2] & pyom.boundary_mask[1:-2,1:-2])[s2], axis=(0,1))
    return east + west + north + south

@pyom_method
def apply_op(pyom, cf, p1, res):
    """
    apply operator A,  res = A *p1
    """
    P1 = np.empty((pyom.nx, pyom.ny, 3,3))
    P1[:,:,0,0] = p1[1:pyom.nx+1, 1:pyom.ny+1]
    P1[:,:,0,1] = p1[1:pyom.nx+1, 2:pyom.ny+2]
    P1[:,:,0,2] = p1[1:pyom.nx+1, 3:pyom.ny+3]
    P1[:,:,1,0] = p1[2:pyom.nx+2, 1:pyom.ny+1]
    P1[:,:,1,1] = p1[2:pyom.nx+2, 2:pyom.ny+2]
    P1[:,:,1,2] = p1[2:pyom.nx+2, 3:pyom.ny+3]
    P1[:,:,2,0] = p1[3:pyom.nx+3, 1:pyom.ny+1]
    P1[:,:,2,1] = p1[3:pyom.nx+3, 2:pyom.ny+2]
    P1[:,:,2,2] = p1[3:pyom.nx+3, 3:pyom.ny+3]
    res[2:pyom.nx+2, 2:pyom.ny+2] = np.sum(cf[2:pyom.nx+2, 2:pyom.ny+2] * P1, axis=(2,3))

@pyom_method
def absmax_sfp(pyom, p1):
    s2 = 0
    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=pyom.js_pe,pyom.je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=pyom.is_pe,pyom.ie_pe
            s2 = max(abs(p1[i,j]*pyom.maskT[i,j,-1]), s2)
            #s2 = max( abs(p1(i,j)), s2 )
    return s2

@pyom_method
def dot_sfp(pyom, p1, p2):
    s2 = 0
    for j in xrange(pyom.js_pe, pyom.je_pe+1): #j=pyom.js_pe,pyom.je_pe
        for i in xrange(pyom.is_pe, pyom.ie_pe+1): #i=pyom.is_pe,pyom.ie_pe
            s2 = s2+p1[i,j]*p2[i,j]*pyom.maskT[i,j,-1]
            #s2 = s2+p1(i,j)*p2(i,j)
    return s2
