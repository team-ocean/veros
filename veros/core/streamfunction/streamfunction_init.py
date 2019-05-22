from loguru import logger

from ... import veros_method, runtime_settings as rs, runtime_state as rst
from ...variables import allocate
from .. import utilities as mainutils
from . import island, utilities


@veros_method(inline=True, dist_safe=False, local_variables=['kbot', 'land_map'])
def get_isleperim(vs):
    logger.debug(' Determining number of land masses')
    vs.land_map[...] = island.isleperim(vs, vs.kbot)
    logger.info(_ascii_map(vs, vs.land_map))
    return int(vs.land_map.max())


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


@veros_method
def streamfunction_init(vs):
    """
    prepare for island integrals
    """
    logger.info('Initializing streamfunction method')

    """
    preprocess land map using MOMs algorithm for B-grid to determine number of islands
    """
    vs.land_map = allocate(vs, ('xt', 'yt'), dtype='int')
    nisle = get_isleperim(vs)

    """
    now that the number of islands is known we can allocate the rest of the variables
    """
    vs.nisle = nisle
    vs.isle = np.arange(1, vs.nisle + 1)
    vs.psin = allocate(vs, ('xu', 'yu', 'isle'))
    vs.dpsin = allocate(vs, ('isle', 'timesteps'))
    vs.line_psin = allocate(vs, ('isle', 'isle'))
    vs.boundary_mask = allocate(vs, ('xt', 'yt', 'isle'), dtype='bool')
    vs.line_dir_south_mask = allocate(vs, ('xt', 'yt', 'isle'), dtype='bool')
    vs.line_dir_north_mask = allocate(vs, ('xt', 'yt', 'isle'), dtype='bool')
    vs.line_dir_east_mask = allocate(vs, ('xt', 'yt', 'isle'), dtype='bool')
    vs.line_dir_west_mask = allocate(vs, ('xt', 'yt', 'isle'), dtype='bool')

    for isle in range(vs.nisle):
        boundary_map = vs.land_map == (isle + 1)

        if vs.enable_cyclic_x:
            vs.line_dir_east_mask[2:-2, 1:-1, isle] = boundary_map[3:-1, 1:-1] & ~boundary_map[3:-1, 2:]
            vs.line_dir_west_mask[2:-2, 1:-1, isle] = boundary_map[2:-2, 2:] & ~boundary_map[2:-2, 1:-1]
            vs.line_dir_south_mask[2:-2, 1:-1, isle] = boundary_map[2:-2, 1:-1] & ~boundary_map[3:-1, 1:-1]
            vs.line_dir_north_mask[2:-2, 1:-1, isle] = boundary_map[3:-1, 2:] & ~boundary_map[2:-2, 2:]
        else:
            vs.line_dir_east_mask[1:-1, 1:-1, isle] = boundary_map[2:, 1:-1] & ~boundary_map[2:, 2:]
            vs.line_dir_west_mask[1:-1, 1:-1, isle] = boundary_map[1:-1, 2:] & ~boundary_map[1:-1, 1:-1]
            vs.line_dir_south_mask[1:-1, 1:-1, isle] = boundary_map[1:-1, 1:-1] & ~boundary_map[2:, 1:-1]
            vs.line_dir_north_mask[1:-1, 1:-1, isle] = boundary_map[2:, 2:] & ~boundary_map[1:-1, 2:]

        vs.boundary_mask[..., isle] = (
            vs.line_dir_east_mask[..., isle]
            | vs.line_dir_west_mask[..., isle]
            | vs.line_dir_north_mask[..., isle]
            | vs.line_dir_south_mask[..., isle]
        )

    vs.linear_solver = _get_solver_class()(vs)

    """
    precalculate time independent boundary components of streamfunction
    """
    forc = allocate(vs, ('xu', 'yu'))

    # initialize with random noise to achieve uniform convergence
    vs.psin[...] = vs.maskZ[..., -1, np.newaxis]# np.random.rand(*vs.psin.shape) * vs.maskZ[..., -1, np.newaxis]

    for isle in range(vs.nisle):
        logger.info(' Solving for boundary contribution by island {:d}'.format(isle))
        vs.linear_solver.solve(vs, forc, vs.psin[:, :, isle],
                               boundary_val=vs.boundary_mask[:, :, isle])

    mainutils.enforce_boundaries(vs, vs.psin)

    """
    precalculate time independent island integrals
    """
    fpx = allocate(vs, ('xu', 'yu', 'isle'))
    fpy = allocate(vs, ('xu', 'yu', 'isle'))

    fpx[1:, 1:, :] = -vs.maskU[1:, 1:, -1, np.newaxis] \
        * (vs.psin[1:, 1:, :] - vs.psin[1:, :-1, :]) \
        / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.hur[1:, 1:, np.newaxis]
    fpy[1:, 1:, ...] = vs.maskV[1:, 1:, -1, np.newaxis] \
        * (vs.psin[1:, 1:, :] - vs.psin[:-1, 1:, :]) \
        / (vs.cosu[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
        * vs.hvr[1:, 1:, np.newaxis]
    vs.line_psin[...] = utilities.line_integrals(vs, fpx, fpy, kind='full')


@veros_method
def _ascii_map(vs, boundary_map):
    map_string = ''
    linewidth = 100
    imt = vs.nx + 4
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
            for j in range(vs.ny + 3, -1, -1):
                map_string += '{:3d} '.format(j) + ''.join([str(boundary_map[istart + i - 2, j] % 10)
                                                            if boundary_map[istart + i - 2, j] >= 0 else '*' for i in range(2, iline + 2)])
                map_string += '\n'
            map_string += ''.join(['{:5d}'.format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += '\n'
            istart = istart + iline
    map_string += '\n'
    return map_string
