from loguru import logger

from veros.core import veros_kernel, runtime_settings as rs, runtime_state as rst
from .. import utilities as mainutils
from . import island, utilities


@veros_kernel(inline=True, dist_safe=False, local_variables=['kbot', 'land_map'])
def get_isleperim(land_map, kbot, nx, ny, enable_cyclic_x):
    logger.debug(' Determining number of land masses')
    land_map[...] = island.isleperim(kbot, enable_cyclic_x)
    logger.info(_ascii_map(nx, ny, land_map))
    return int(land_map.max()), land_map


def _get_solver_class():
    ls = rs.linear_solver

    def _get_best_solver():
        if rst.proc_num > 1:
            try:
                from .solvers.petsc import PETScSolver
            except ImportError:
                logger.warning('PETSc linear solver not available, falling back to pyAMG')
            else:
                return PETScSolver

        try:
            from .solvers.pyamg import PyAMGSolver
        except ImportError:
            logger.warning('pyAMG linear solver not available, falling back to SciPy')
        else:
            return PyAMGSolver

        from .solvers.scipy import SciPySolver
        return SciPySolver

    if ls == 'best':
        return _get_best_solver()
    elif ls == 'petsc':
        from .solvers.petsc import PETScSolver
        return PETScSolver
    elif ls == 'pyamg':
        from .solvers.pyamg import PyAMGSolver
        return PyAMGSolver
    elif ls == 'scipy':
        from .solvers.scipy import SciPySolver
        return SciPySolver

    raise ValueError('unrecognized linear solver %s' % ls)


@veros_kernel
def streamfunction_init(land_map, nisle, kbot, nx, ny, isle, psin, dpsin, line_psin, boundary_mask,
                        line_dir_south_mask, line_dir_north_mask, line_dir_east_mask,
                        line_dir_west_mask, linear_solver, dxt, dyt, hur, hvr, maskZ,
                        maskU, maskV, cosu, dxu, dyu, cost, default_float_type,
                        enable_cyclic_x):
    """
    prepare for island integrals
    """
    logger.info('Initializing streamfunction method')

    """
    preprocess land map using MOMs algorithm for B-grid to determine number of islands
    """
    land_map = np.zeros_like(kbot, dtype='int')
    nisle, land_map = get_isleperim(land_map, kbot, nx, ny, enable_cyclic_x)

    """
    now that the number of islands is known we can allocate the rest of the variables
    """
    isle = np.arange(1, nisle + 1)
    psin = np.zeros((nx, ny, isle), dtype=default_float_type)
    dpsin = np.zeros((nisle, 3), dtype=default_float_type)          # !!! CHANGE ALLOCATION !!!
    line_psin = np.zeros((nisle, nisle), dtype=default_float_type)
    boundary_mask = np.zeros((nx, ny, isle), dtype='bool')
    line_dir_south_mask = np.zeros((nx, ny, isle), dtype='bool')
    line_dir_north_mask = np.zeros((nx, ny, isle), dtype='bool')
    line_dir_east_mask = np.zeros((nx, ny, isle), dtype='bool')
    line_dir_west_mask = np.zeros((nx, ny, isle), dtype='bool')

    for isle in range(nisle):
        boundary_map = land_map == (isle + 1)

        if enable_cyclic_x:
            line_dir_east_mask[2:-2, 1:-1, isle] = boundary_map[3:-1, 1:-1] & ~boundary_map[3:-1, 2:]
            line_dir_west_mask[2:-2, 1:-1, isle] = boundary_map[2:-2, 2:] & ~boundary_map[2:-2, 1:-1]
            line_dir_south_mask[2:-2, 1:-1, isle] = boundary_map[2:-2, 1:-1] & ~boundary_map[3:-1, 1:-1]
            line_dir_north_mask[2:-2, 1:-1, isle] = boundary_map[3:-1, 2:] & ~boundary_map[2:-2, 2:]
        else:
            line_dir_east_mask[1:-1, 1:-1, isle] = boundary_map[2:, 1:-1] & ~boundary_map[2:, 2:]
            line_dir_west_mask[1:-1, 1:-1, isle] = boundary_map[1:-1, 2:] & ~boundary_map[1:-1, 1:-1]
            line_dir_south_mask[1:-1, 1:-1, isle] = boundary_map[1:-1, 1:-1] & ~boundary_map[2:, 1:-1]
            line_dir_north_mask[1:-1, 1:-1, isle] = boundary_map[2:, 2:] & ~boundary_map[1:-1, 2:]

        boundary_mask[..., isle] = (
            line_dir_east_mask[..., isle]
            | line_dir_west_mask[..., isle]
            | line_dir_north_mask[..., isle]
            | line_dir_south_mask[..., isle]
        )

    linear_solver = _get_solver_class()

    """
    precalculate time independent boundary components of streamfunction
    """
    forc = np.zeros((nx, ny), dtype=default_float_type)

    # initialize with random noise to achieve uniform convergence
    psin[...] = maskZ[..., -1, np.newaxis]  # np.random.rand(*psin.shape) * maskZ[..., -1, np.newaxis]

    for isle in range(nisle):
        logger.info(' Solving for boundary contribution by island {:d}'.format(isle))
        linear_solver.solve(forc, psin[:, :, isle],
                            boundary_val=boundary_mask[:, :, isle])

    mainutils.enforce_boundaries(psin, enable_cyclic_x)

    """
    precalculate time independent island integrals
    """
    fpx = np.zeros((nx, ny, isle), dtype=default_float_type)
    fpy = np.zeros((nx, ny, isle), dtype=default_float_type)

    fpx[1:, 1:, :] = -maskU[1:, 1:, -1, np.newaxis] \
        * (psin[1:, 1:, :] - psin[1:, :-1, :]) \
        / dyt[np.newaxis, 1:, np.newaxis] * hur[1:, 1:, np.newaxis]
    fpy[1:, 1:, ...] = maskV[1:, 1:, -1, np.newaxis] \
        * (psin[1:, 1:, :] - psin[:-1, 1:, :]) \
        / (cosu[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis]) \
        * hvr[1:, 1:, np.newaxis]
    line_psin[...] = utilities.line_integrals(dxu, dyu, cost, line_dir_east_mask,
        line_dir_west_mask, line_dir_north_mask, line_dir_south_mask, boundary_mask,
        nisle, uloc=fpx, vloc=fpy, kind='full')

    return psin, dpsin, line_psin, land_map, boundary_mask, line_dir_south_mask,\
        line_dir_north_mask, line_dir_east_mask, line_dir_west_mask, linear_solver


@veros_kernel
def _ascii_map(nx, ny, boundary_map):
    map_string = ''
    linewidth = 100
    imt = nx + 4
    iremain = imt
    istart = 0
    map_string += '\n'
    map_string += ' ' * (5 + min(linewidth, imt) // 2 - 13) + 'Land mass and perimeter'
    map_string += '\n'
    for isweep in range(1, imt // linewidth + 2):
        iline = min(iremain, linewidth)
        iremain = iremain - iline
        if iline > 0:
            map_string += '\n'
            map_string += ''.join(['{:5d}'.format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += '\n'
            for j in range(ny + 3, -1, -1):
                map_string += '{:3d} '.format(j) + ''.join([str(boundary_map[istart + i - 2, j] % 10)
                                                            if boundary_map[istart + i - 2, j] >= 0 else '*' for i in range(2, iline + 2)])
                map_string += '\n'
            map_string += ''.join(['{:5d}'.format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += '\n'
            istart = istart + iline
    map_string += '\n'
    return map_string
