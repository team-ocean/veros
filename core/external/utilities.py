from ... import veros_method


@veros_method
def line_integrals(vs, uloc, vloc, kind="same"):
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

    east = vloc[1:-2, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 2:-1, np.newaxis]
    west = -vloc[2:-1, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 1:-2, np.newaxis]
    north = vloc[1:-2, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 1:-2, np.newaxis]
    south = -vloc[2:-1, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 2:-1, np.newaxis]
    east = np.sum(east[s1] * (vs.line_dir_east_mask[1:-2, 1:-2] &
                              vs.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))
    west = np.sum(west[s1] * (vs.line_dir_west_mask[1:-2, 1:-2] &
                              vs.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))
    north = np.sum(north[s1] * (vs.line_dir_north_mask[1:-2, 1:-2]
                                & vs.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))
    south = np.sum(south[s1] * (vs.line_dir_south_mask[1:-2, 1:-2]
                                & vs.boundary_mask[1:-2, 1:-2])[s2], axis=(0, 1))

    return east + west + north + south
