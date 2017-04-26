from ... import veros_method


@veros_method
def line_integrals(veros, uloc, vloc, kind="same"):
    """
    calculate line integrals along all islands

    Arguments:
        kind: "same" calculates only line integral contributions of an island with itself,
               while "full" calculates all possible pairings between all islands.
    """
    if kind == "same":
        s1 = s2 = (slice(None), slice(None), slice(None))
    elif kind == "full":
        s1 = (slice(None), slice(None), np.newaxis, slice(None))
        s2 = (slice(None), slice(None), slice(None), np.newaxis)
    else:
        raise ValueError("kind must be 'same' or 'full'")

    east = vloc[1:-2, 1:-2, :] * veros.dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * veros.dxu[1:-2, np.newaxis, np.newaxis] \
        * veros.cost[np.newaxis, 2:-1, np.newaxis]
    west = -vloc[2:-1, 1:-2, :] * veros.dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * veros.dxu[1:-2, np.newaxis, np.newaxis] \
        * veros.cost[np.newaxis, 1:-2, np.newaxis]
    north = vloc[1:-2, 1:-2, :] * veros.dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * veros.dxu[1:-2, np.newaxis, np.newaxis] \
        * veros.cost[np.newaxis, 1:-2, np.newaxis]
    south = -vloc[2:-1, 1:-2, :] * veros.dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * veros.dxu[1:-2, np.newaxis, np.newaxis] \
        * veros.cost[np.newaxis, 2:-1, np.newaxis]
    east = np.sum(east[s1] * (veros.line_dir_east_mask[1:-2, 1:-2] &
                              veros.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))
    west = np.sum(west[s1] * (veros.line_dir_west_mask[1:-2, 1:-2] &
                              veros.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))
    north = np.sum(north[s1] * (veros.line_dir_north_mask[1:-2, 1:-2]
                                & veros.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))
    south = np.sum(south[s1] * (veros.line_dir_south_mask[1:-2, 1:-2]
                                & veros.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))

    return east + west + north + south


@veros_method
def apply_op(veros, cf, p1, res):
    """
    apply operator A,  res = A *p1
    """
    P1 = np.empty((veros.nx, veros.ny, 3, 3))
    P1[:, :, 0, 0] = p1[1:veros.nx + 1, 1:veros.ny + 1]
    P1[:, :, 0, 1] = p1[1:veros.nx + 1, 2:veros.ny + 2]
    P1[:, :, 0, 2] = p1[1:veros.nx + 1, 3:veros.ny + 3]
    P1[:, :, 1, 0] = p1[2:veros.nx + 2, 1:veros.ny + 1]
    P1[:, :, 1, 1] = p1[2:veros.nx + 2, 2:veros.ny + 2]
    P1[:, :, 1, 2] = p1[2:veros.nx + 2, 3:veros.ny + 3]
    P1[:, :, 2, 0] = p1[3:veros.nx + 3, 1:veros.ny + 1]
    P1[:, :, 2, 1] = p1[3:veros.nx + 3, 2:veros.ny + 2]
    P1[:, :, 2, 2] = p1[3:veros.nx + 3, 3:veros.ny + 3]
    res[2:veros.nx + 2, 2:veros.ny +
        2] = np.sum(cf[2:veros.nx + 2, 2:veros.ny + 2] * P1, axis=(2, 3))


@veros_method
def absmax_sfp(veros, p1):
    s2 = 0
    for j in xrange(veros.js_pe, veros.je_pe + 1):
        for i in xrange(veros.is_pe, veros.ie_pe + 1):
            s2 = max(abs(p1[i, j] * veros.maskT[i, j, -1]), s2)
    return s2


@veros_method
def dot_sfp(veros, p1, p2):
    s2 = 0
    for j in xrange(veros.js_pe, veros.je_pe + 1):
        for i in xrange(veros.is_pe, veros.ie_pe + 1):
            s2 = s2 + p1[i, j] * p2[i, j] * veros.maskT[i, j, -1]
    return s2
