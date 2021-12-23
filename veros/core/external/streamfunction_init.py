from veros import logger, veros_kernel, veros_routine, KernelOutput
from veros.variables import allocate
from veros.distributed import global_max
from veros.core import utilities as mainutils
from veros.core.operators import numpy as npx, update, at
from veros.core.external import island, line_integrals, solve_stream
from veros.core.external.solvers import get_linear_solver


@veros_routine
def get_isleperim(state):
    """
    preprocess land map using MOMs algorithm for B-grid to determine number of islands
    """
    from veros.state import resize_dimension

    vs = state.variables

    island.isleperim(state)

    # now that we know the number of islands we can resize
    # all arrays depending on that
    nisle = int(global_max(npx.max(vs.land_map)))
    resize_dimension(state, "isle", nisle)
    vs.isle = npx.arange(nisle)


@veros_routine
def streamfunction_init(state):
    """
    prepare for island integrals
    """
    vs = state.variables
    settings = state.settings

    logger.info("Initializing streamfunction method")

    get_isleperim(state)

    vs.update(boundary_masks(state))

    # populate linear solver cache
    linear_solver = get_linear_solver(state)

    """
    precalculate time independent boundary components of streamfunction
    """
    forc = allocate(state.dimensions, ("xt", "yt"))

    vs.psin = update(vs.psin, at[...], vs.maskZ[..., -1, npx.newaxis])

    for isle in range(state.dimensions["isle"]):
        logger.info(f" Solving for boundary contribution by island {isle:d}")
        isle_boundary = (
            vs.line_dir_east_mask[..., isle]
            | vs.line_dir_west_mask[..., isle]
            | vs.line_dir_north_mask[..., isle]
            | vs.line_dir_south_mask[..., isle]
        )
        isle_sol = linear_solver.solve(state, forc, vs.psin[:, :, isle], boundary_val=isle_boundary)
        vs.psin = update(vs.psin, at[:, :, isle], isle_sol)

    vs.psin = mainutils.enforce_boundaries(vs.psin, settings.enable_cyclic_x)

    line_psin_out = island_integrals(state)
    vs.update(line_psin_out)

    """
    take care of initial velocity
    """

    # transfer initial velocity to tendency
    vs.du = update(vs.du, at[..., vs.tau], vs.u[..., vs.tau] / settings.dt_mom / (1.5 + settings.AB_eps))
    vs.dv = update(vs.dv, at[..., vs.tau], vs.v[..., vs.tau] / settings.dt_mom / (1.5 + settings.AB_eps))
    vs.u = update(vs.u, at[...], 0)
    vs.v = update(vs.v, at[...], 0)

    # run streamfunction solver to determine initial barotropic and baroclinic modes
    solve_stream.solve_streamfunction(state)

    vs.psi = update(vs.psi, at[...], vs.psi[..., vs.taup1, npx.newaxis])
    vs.u = update(
        vs.u, at[...], mainutils.enforce_boundaries(vs.u[..., vs.taup1, npx.newaxis], settings.enable_cyclic_x)
    )
    vs.v = update(
        vs.v, at[...], mainutils.enforce_boundaries(vs.v[..., vs.taup1, npx.newaxis], settings.enable_cyclic_x)
    )
    vs.du = update(vs.du, at[..., vs.tau], 0)
    vs.dv = update(vs.dv, at[..., vs.tau], 0)


@veros_kernel
def island_integrals(state):
    """
    precalculate time independent island integrals
    """
    vs = state.variables

    uloc = allocate(state.dimensions, ("xt", "yt", "isle"))
    vloc = allocate(state.dimensions, ("xt", "yt", "isle"))

    uloc = update(
        uloc,
        at[1:, 1:, :],
        -(vs.psin[1:, 1:, :] - vs.psin[1:, :-1, :])
        * vs.maskU[1:, 1:, -1, npx.newaxis]
        / vs.dyt[npx.newaxis, 1:, npx.newaxis]
        * vs.hur[1:, 1:, npx.newaxis],
    )

    vloc = update(
        vloc,
        at[1:, 1:, ...],
        (vs.psin[1:, 1:, :] - vs.psin[:-1, 1:, :])
        * vs.maskV[1:, 1:, -1, npx.newaxis]
        / (vs.cosu[npx.newaxis, 1:, npx.newaxis] * vs.dxt[1:, npx.newaxis, npx.newaxis])
        * vs.hvr[1:, 1:, npx.newaxis],
    )

    vs.line_psin = line_integrals.line_integrals(state, uloc=uloc, vloc=vloc, kind="full")
    return KernelOutput(line_psin=vs.line_psin)


@veros_kernel
def boundary_masks(state):
    """
    now that the number of islands is known we can allocate the rest of the variables
    """
    vs = state.variables
    settings = state.settings

    boundary_map = vs.land_map[..., npx.newaxis] == npx.arange(1, state.dimensions["isle"] + 1)

    if settings.enable_cyclic_x:
        vs.line_dir_east_mask = update(
            vs.line_dir_east_mask, at[2:-2, 1:-1], boundary_map[3:-1, 1:-1] & ~boundary_map[3:-1, 2:]
        )
        vs.line_dir_west_mask = update(
            vs.line_dir_west_mask, at[2:-2, 1:-1], boundary_map[2:-2, 2:] & ~boundary_map[2:-2, 1:-1]
        )
        vs.line_dir_south_mask = update(
            vs.line_dir_south_mask, at[2:-2, 1:-1], boundary_map[2:-2, 1:-1] & ~boundary_map[3:-1, 1:-1]
        )
        vs.line_dir_north_mask = update(
            vs.line_dir_north_mask, at[2:-2, 1:-1], boundary_map[3:-1, 2:] & ~boundary_map[2:-2, 2:]
        )
    else:
        vs.line_dir_east_mask = update(
            vs.line_dir_east_mask, at[1:-1, 1:-1], boundary_map[2:, 1:-1] & ~boundary_map[2:, 2:]
        )
        vs.line_dir_west_mask = update(
            vs.line_dir_west_mask, at[1:-1, 1:-1], boundary_map[1:-1, 2:] & ~boundary_map[1:-1, 1:-1]
        )
        vs.line_dir_south_mask = update(
            vs.line_dir_south_mask, at[1:-1, 1:-1], boundary_map[1:-1, 1:-1] & ~boundary_map[2:, 1:-1]
        )
        vs.line_dir_north_mask = update(
            vs.line_dir_north_mask, at[1:-1, 1:-1], boundary_map[2:, 2:] & ~boundary_map[1:-1, 2:]
        )

    vs.isle_boundary_mask = ~npx.any(
        vs.line_dir_east_mask | vs.line_dir_west_mask | vs.line_dir_south_mask | vs.line_dir_north_mask, axis=2
    )

    return KernelOutput(
        isle_boundary_mask=vs.isle_boundary_mask,
        line_dir_east_mask=vs.line_dir_east_mask,
        line_dir_west_mask=vs.line_dir_west_mask,
        line_dir_south_mask=vs.line_dir_south_mask,
        line_dir_north_mask=vs.line_dir_north_mask,
    )
