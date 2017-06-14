import logging

from ... import veros_method
from .. import cyclic
from . import island, utilities, solve_poisson


@veros_method
def streamfunction_init(vs):
    """
    prepare for island integrals
    """
    logging.info("Initializing streamfunction method")

    """
    preprocess land map using MOMs algorithm for B-grid to determine number of islands
    """
    logging.info(" determining number of land masses")
    allmap = island.isleperim(vs, vs.kbot, verbose=vs.verbose_island_routines)
    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(allmap)
    logging.info(_ascii_map(vs, allmap))

    """
    now that the number of islands is known we can allocate the rest of the variables
    """
    vs.nisle = int(allmap.max())
    vs.isle = np.arange(vs.nisle) + 1
    vs.psin = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=vs.default_float_type)
    vs.dpsin = np.zeros((vs.nisle, 3), dtype=vs.default_float_type)
    vs.line_psin = np.zeros((vs.nisle, vs.nisle), dtype=vs.default_float_type)
    vs.boundary_mask = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=np.bool)
    vs.line_dir_south_mask = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=np.bool)
    vs.line_dir_north_mask = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=np.bool)
    vs.line_dir_east_mask = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=np.bool)
    vs.line_dir_west_mask = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=np.bool)

    if vs.backend_name == "bohrium":
        vs.boundary_mask = vs.boundary_mask.copy2numpy()
        vs.line_dir_south_mask = vs.line_dir_south_mask.copy2numpy()
        vs.line_dir_north_mask = vs.line_dir_north_mask.copy2numpy()
        vs.line_dir_east_mask = vs.line_dir_east_mask.copy2numpy()
        vs.line_dir_west_mask = vs.line_dir_west_mask.copy2numpy()

    for isle in range(vs.nisle):
        _print_verbose(vs, " ------------------------")
        _print_verbose(vs, " processing land mass #{:d}".format(isle))
        _print_verbose(vs, " ------------------------")

        """
        land map for island number isle: 1 is land, -1 is perimeter, 0 is ocean
        """
        boundary_map = island.isleperim(vs, allmap != isle + 1)
        if vs.backend_name == "bohrium":
            boundary_map = boundary_map.copy2numpy()
        _print_verbose(vs, _ascii_map(vs, boundary_map))

        """
        find a starting point
        """
        n = 0
        # avoid starting close to cyclic bondaries
        cont, ij, Dir, startPos = _avoid_cyclic_boundaries(
            vs, boundary_map, isle, n, (vs.nx // 2 + 1, vs.nx + 2))
        if not cont:
            cont, ij, Dir, startPos = _avoid_cyclic_boundaries(
                vs, boundary_map, isle, n, (vs.nx // 2, -1, -1))
            if not cont:
                raise RuntimeError("found no starting point for line integral")

        _print_verbose(vs, " starting point of line integral is {!r}".format(startPos))
        _print_verbose(vs, " starting direction is {!r}".format(Dir))

        """
        now find connecting lines
        """
        n = 1
        vs.boundary_mask[ij[0], ij[1], isle] = 1
        cont = True
        while cont:
            """
            consider map in front of line direction and to the right and decide where to go
            """
            if Dir[0] == 0 and Dir[1] == 1:  # north
                ijp = [ij[0], ij[1] + 1]
                ijp_right = [ij[0] + 1, ij[1] + 1]
            elif Dir[0] == -1 and Dir[1] == 0:  # west
                ijp = [ij[0], ij[1]]
                ijp_right = [ij[0], ij[1] + 1]
            elif Dir[0] == 0 and Dir[1] == -1:  # south
                ijp = [ij[0] + 1, ij[1]]
                ijp_right = [ij[0], ij[1]]
            elif Dir[0] == 1 and Dir[1] == 0:  # east
                ijp = [ij[0] + 1, ij[1] + 1]
                ijp_right = [ij[0] + 1, ij[1]]

            """
            4 cases are possible
            """

            _print_verbose(vs, " ")
            _print_verbose(vs, " position is {!r}".format(ij))
            _print_verbose(vs, " direction is {!r}".format(Dir))
            _print_verbose(vs, " map ahead is {} {}"
                          .format(boundary_map[ijp[0], ijp[1]], boundary_map[ijp_right[0], ijp_right[1]]))

            if boundary_map[ijp[0], ijp[1]] == -1 and boundary_map[ijp_right[0], ijp_right[1]] == 1:
                _print_verbose(vs, " go forward")
            elif boundary_map[ijp[0], ijp[1]] == -1 and boundary_map[ijp_right[0], ijp_right[1]] == -1:
                _print_verbose(vs, " turn right")
                Dir = [Dir[1], -Dir[0]]
            elif boundary_map[ijp[0], ijp[1]] == 1 and boundary_map[ijp_right[0], ijp_right[1]] == 1:
                _print_verbose(vs, " turn left")
                Dir = [-Dir[1], Dir[0]]
            elif boundary_map[ijp[0], ijp[1]] == 1 and boundary_map[ijp_right[0], ijp_right[1]] == -1:
                _print_verbose(vs, " turn left")
                Dir = [-Dir[1], Dir[0]]
            else:
                _print_verbose(vs, " map ahead is {} {}"
                             .format(boundary_map[ijp[0], ijp[1]], boundary_map[ijp_right[0], ijp_right[1]]))
                raise RuntimeError("unknown situation or lost track")

            """
            go forward in direction
            """
            if Dir[0] == 0 and Dir[1] == 1:  # north
                vs.line_dir_north_mask[ij[0], ij[1], isle] = 1
            elif Dir[0] == -1 and Dir[1] == 0:  # west
                vs.line_dir_west_mask[ij[0], ij[1], isle] = 1
            elif Dir[0] == 0 and Dir[1] == -1:  # south
                vs.line_dir_south_mask[ij[0], ij[1], isle] = 1
            elif Dir[0] == 1 and Dir[1] == 0:  # east
                vs.line_dir_east_mask[ij[0], ij[1], isle] = 1
            ij = [ij[0] + Dir[0], ij[1] + Dir[1]]
            if startPos[0] == ij[0] and startPos[1] == ij[1]:
                cont = False

            """
            account for cyclic boundary conditions
            """
            if vs.enable_cyclic_x and Dir[0] == 1 and Dir[1] == 0 and ij[0] > vs.nx + 1:
                _print_verbose(vs, " shifting to western cyclic boundary")
                ij[0] -= vs.nx
            if vs.enable_cyclic_x and Dir[0] == -1 and Dir[1] == 0 and ij[0] < 2:
                _print_verbose(vs, " shifting to eastern cyclic boundary")
                ij[0] += vs.nx
            if startPos[0] == ij[0] and startPos[1] == ij[1]:
                cont = False

            if cont:
                n += 1
                vs.boundary_mask[ij[0], ij[1], isle] = 1

        _print_verbose(vs, " number of points is {:d}".format(n + 1))
        _print_verbose(vs, " ")
        _print_verbose(vs, " Positions:")
        _print_verbose(vs, " boundary: {!r}".format(vs.boundary_mask[..., isle]))

    if vs.backend_name == "bohrium":
        vs.boundary_mask = np.asarray(vs.boundary_mask)
        vs.line_dir_south_mask = np.asarray(vs.line_dir_south_mask)
        vs.line_dir_north_mask = np.asarray(vs.line_dir_north_mask)
        vs.line_dir_east_mask = np.asarray(vs.line_dir_east_mask)
        vs.line_dir_west_mask = np.asarray(vs.line_dir_west_mask)

    solve_poisson.initialize_solver(vs)

    """
    precalculate time independent boundary components of streamfunction
    """
    forc = np.zeros((vs.nx+4, vs.ny+4), dtype=vs.default_float_type)

    # initialize with random noise to achieve uniform convergence
    vs.psin[...] = np.random.rand(*vs.psin.shape) * vs.maskZ[..., -1, np.newaxis]

    for isle in range(vs.nisle):
        logging.info(" solving for boundary contribution by island {:d}".format(isle))
        solve_poisson.solve(vs, forc, vs.psin[:, :, isle],
                            boundary_val=vs.boundary_mask[:, :, isle])

    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(vs.psin)

    """
    precalculate time independent island integrals
    """
    fpx = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=vs.default_float_type)
    fpy = np.zeros((vs.nx + 4, vs.ny + 4, vs.nisle), dtype=vs.default_float_type)

    fpx[1:, 1:, :] = -vs.maskU[1:, 1:, -1, np.newaxis] \
        * (vs.psin[1:, 1:, :] - vs.psin[1:, :-1, :]) \
        / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.hur[1:, 1:, np.newaxis]
    fpy[1:, 1:, ...] = vs.maskV[1:, 1:, -1, np.newaxis] \
        * (vs.psin[1:, 1:, :] - vs.psin[:-1, 1:, :]) \
        / (vs.cosu[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
        * vs.hvr[1:, 1:, np.newaxis]
    vs.line_psin[...] = utilities.line_integrals(vs, fpx, fpy, kind="full")


@veros_method
def _avoid_cyclic_boundaries(vs, boundary_map, isle, n, x_range):
    for i in range(*x_range):
        for j in range(1, vs.ny + 2):
            if boundary_map[i, j] == 1 and boundary_map[i, j + 1] == -1:
                # initial direction is eastward, we come from the west
                cont = True
                Dir = [1, 0]
                vs.line_dir_east_mask[i - 1, j, isle] = 1
                vs.boundary_mask[i - 1, j, isle] = 1
                return cont, (i, j), Dir, (i - 1, j)
            if boundary_map[i, j] == -1 and boundary_map[i, j + 1] == 1:
                # initial direction is westward, we come from the east
                cont = True
                Dir = [-1, 0]
                vs.line_dir_west_mask[i, j, isle] = 1
                vs.boundary_mask[i, j, isle] = 1
                return cont, (i - 1, j), Dir, (i, j)
    return False, None, [0, 0], [0, 0]

def _print_verbose(vs, message):
    if vs.verbose_island_routines:
        logging.info(message)

@veros_method
def _ascii_map(vs, boundary_map):
    map_string = ""
    linewidth = 125
    imt = vs.nx + 4
    iremain = imt
    istart = 0
    map_string += "\n"
    map_string += " " * (5 + min(linewidth, imt) // 2 - 13) + "Land mass and perimeter"
    map_string += "\n"
    for isweep in range(1, imt // linewidth + 2):
        iline = min(iremain, linewidth)
        iremain = iremain - iline
        if iline > 0:
            map_string += "\n"
            map_string += "".join(["{:5d}".format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += "\n"
            for j in range(vs.ny + 3, -1, -1):
                map_string += "{:3d} ".format(j) + "".join([str(int(_mod10(boundary_map[istart + i - 2, j]))) if _mod10(
                    boundary_map[istart + i - 2, j]) >= 0 else "*" for i in range(2, iline + 2)])
                map_string += "\n"
            map_string += "".join(["{:5d}".format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += "\n"
            istart = istart + iline
    map_string += "\n"
    return map_string


def _mod10(m):
    return m % 10 if m > 0 else m
