from veros.core.operators import update, at, numpy as npx
from veros.variables import allocate


def assemble_poisson_matrix(state):
    if state.settings.enable_streamfunction:
        return assemble_streamfunction_matrix(state)
    else:
        return assemble_pressure_matrix(state)


def assemble_pressure_matrix(state):
    main_diag = allocate(state.dimensions, ("xu", "yu"), fill=1)
    east_diag, west_diag, north_diag, south_diag = (allocate(state.dimensions, ("xu", "yu")) for _ in range(4))

    vs = state.variables
    settings = state.settings

    maskM = vs.maskT[:, :, -1]

    mp_i = maskM[2:-2, 2:-2] * maskM[3:-1, 2:-2]
    mm_i = maskM[2:-2, 2:-2] * maskM[1:-3, 2:-2]

    mp_j = maskM[2:-2, 2:-2] * maskM[2:-2, 3:-1]
    mm_j = maskM[2:-2, 2:-2] * maskM[2:-2, 1:-3]

    main_diag = update(
        main_diag,
        at[2:-2, 2:-2],
        -1
        * mp_i
        * vs.hu[2:-2, 2:-2]
        / vs.dxu[2:-2, npx.newaxis]
        / vs.dxt[2:-2, npx.newaxis]
        / vs.cost[npx.newaxis, 2:-2] ** 2
        - 1
        * mm_i
        * vs.hu[1:-3, 2:-2]
        / vs.dxu[1:-3, npx.newaxis]
        / vs.dxt[2:-2, npx.newaxis]
        / vs.cost[npx.newaxis, 2:-2] ** 2
        - 1
        * mp_j
        * vs.hv[2:-2, 2:-2]
        / vs.dyu[npx.newaxis, 2:-2]
        / vs.dyt[npx.newaxis, 2:-2]
        * vs.cosu[npx.newaxis, 2:-2]
        / vs.cost[npx.newaxis, 2:-2]
        - 1
        * mm_j
        * vs.hv[2:-2, 1:-3]
        / vs.dyu[npx.newaxis, 1:-3]
        / vs.dyt[npx.newaxis, 2:-2]
        * vs.cosu[npx.newaxis, 1:-3]
        / vs.cost[npx.newaxis, 2:-2]
        # free surface
        - 1.0 / (settings.grav * settings.dt_mom * settings.dt_tracer) * maskM[2:-2, 2:-2],
    )

    east_diag = update(
        east_diag,
        at[2:-2, 2:-2],
        mp_i
        * vs.hu[2:-2, 2:-2]
        / vs.dxu[2:-2, npx.newaxis]
        / vs.dxt[2:-2, npx.newaxis]
        / vs.cost[npx.newaxis, 2:-2] ** 2,
    )

    west_diag = update(
        west_diag,
        at[2:-2, 2:-2],
        mm_i
        * vs.hu[1:-3, 2:-2]
        / vs.dxu[1:-3, npx.newaxis]
        / vs.dxt[2:-2, npx.newaxis]
        / vs.cost[npx.newaxis, 2:-2] ** 2,
    )

    north_diag = update(
        north_diag,
        at[2:-2, 2:-2],
        mp_j
        * vs.hv[2:-2, 2:-2]
        / vs.dyu[npx.newaxis, 2:-2]
        / vs.dyt[npx.newaxis, 2:-2]
        * vs.cosu[npx.newaxis, 2:-2]
        / vs.cost[npx.newaxis, 2:-2],
    )

    south_diag = update(
        south_diag,
        at[2:-2, 2:-2],
        mm_j
        * vs.hv[2:-2, 1:-3]
        / vs.dyu[npx.newaxis, 1:-3]
        / vs.dyt[npx.newaxis, 2:-2]
        * vs.cosu[npx.newaxis, 1:-3]
        / vs.cost[npx.newaxis, 2:-2],
    )
    main_diag = main_diag * maskM
    main_diag = npx.where(npx.abs(main_diag) == 0.0, 1, main_diag)

    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    diags = [main_diag, east_diag, west_diag, north_diag, south_diag]

    return diags, offsets, maskM


def assemble_streamfunction_matrix(state):
    vs = state.variables

    # assemble diagonals
    main_diag = allocate(state.dimensions, ("xu", "yu"), fill=1)
    east_diag, west_diag, north_diag, south_diag = (allocate(state.dimensions, ("xu", "yu")) for _ in range(4))
    main_diag = update(
        main_diag,
        at[2:-2, 2:-2],
        -vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, npx.newaxis] / vs.dxt[3:-1, npx.newaxis] / vs.cosu[npx.newaxis, 2:-2] ** 2
        - vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, npx.newaxis] / vs.dxt[2:-2, npx.newaxis] / vs.cosu[npx.newaxis, 2:-2] ** 2
        - vs.hur[2:-2, 2:-2]
        / vs.dyu[npx.newaxis, 2:-2]
        / vs.dyt[npx.newaxis, 2:-2]
        * vs.cost[npx.newaxis, 2:-2]
        / vs.cosu[npx.newaxis, 2:-2]
        - vs.hur[2:-2, 3:-1]
        / vs.dyu[npx.newaxis, 2:-2]
        / vs.dyt[npx.newaxis, 3:-1]
        * vs.cost[npx.newaxis, 3:-1]
        / vs.cosu[npx.newaxis, 2:-2],
    )
    east_diag = update(
        east_diag,
        at[2:-2, 2:-2],
        vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, npx.newaxis] / vs.dxt[3:-1, npx.newaxis] / vs.cosu[npx.newaxis, 2:-2] ** 2,
    )
    west_diag = update(
        west_diag,
        at[2:-2, 2:-2],
        vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, npx.newaxis] / vs.dxt[2:-2, npx.newaxis] / vs.cosu[npx.newaxis, 2:-2] ** 2,
    )
    north_diag = update(
        north_diag,
        at[2:-2, 2:-2],
        vs.hur[2:-2, 3:-1]
        / vs.dyu[npx.newaxis, 2:-2]
        / vs.dyt[npx.newaxis, 3:-1]
        * vs.cost[npx.newaxis, 3:-1]
        / vs.cosu[npx.newaxis, 2:-2],
    )
    south_diag = update(
        south_diag,
        at[2:-2, 2:-2],
        vs.hur[2:-2, 2:-2]
        / vs.dyu[npx.newaxis, 2:-2]
        / vs.dyt[npx.newaxis, 2:-2]
        * vs.cost[npx.newaxis, 2:-2]
        / vs.cosu[npx.newaxis, 2:-2],
    )

    main_diag = main_diag * vs.isle_boundary_mask
    main_diag = npx.where(main_diag == 0.0, 1.0, main_diag)

    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    diags = [
        main_diag,
        east_diag * vs.isle_boundary_mask,
        west_diag * vs.isle_boundary_mask,
        north_diag * vs.isle_boundary_mask,
        south_diag * vs.isle_boundary_mask,
    ]

    return diags, offsets, vs.isle_boundary_mask
