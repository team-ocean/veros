from veros.core.operators import numpy as npx

from veros import veros_kernel, runtime_state
from veros.distributed import global_sum
from veros.core.operators import update, at, for_loop


@veros_kernel(static_args=("kind"))
def line_integrals(state, uloc, vloc, kind="same"):
    """
    calculate line integrals along all islands

    Arguments:
        kind: 'same' calculates only line integral contributions of an island with itself,
               while 'full' calculates all possible pairings between all islands.
    """
    vs = state.variables
    nisle = state.dimensions["isle"]

    ipx, ipy = runtime_state.proc_idx

    if ipx == 0:
        i = slice(1, -2)
        ip1 = slice(2, -1)
    else:
        i = slice(2, -2)
        ip1 = slice(3, -1)

    if ipy == 0:
        j = slice(1, -2)
        jp1 = slice(2, -1)
    else:
        j = slice(2, -2)
        jp1 = slice(3, -1)

    east = (
        vloc[i, j, :] * vs.dyu[npx.newaxis, j, npx.newaxis]
        + uloc[i, jp1, :] * vs.dxu[i, npx.newaxis, npx.newaxis] * vs.cost[npx.newaxis, jp1, npx.newaxis]
    )
    west = (
        -vloc[ip1, j, :] * vs.dyu[npx.newaxis, j, npx.newaxis]
        - uloc[i, j, :] * vs.dxu[i, npx.newaxis, npx.newaxis] * vs.cost[npx.newaxis, j, npx.newaxis]
    )
    north = (
        vloc[i, j, :] * vs.dyu[npx.newaxis, j, npx.newaxis]
        - uloc[i, j, :] * vs.dxu[i, npx.newaxis, npx.newaxis] * vs.cost[npx.newaxis, j, npx.newaxis]
    )
    south = (
        -vloc[ip1, j, :] * vs.dyu[npx.newaxis, j, npx.newaxis]
        + uloc[i, jp1, :] * vs.dxu[i, npx.newaxis, npx.newaxis] * vs.cost[npx.newaxis, jp1, npx.newaxis]
    )

    if kind == "same":
        east = npx.sum(east * (vs.line_dir_east_mask[i, j] & vs.boundary_mask[i, j]), axis=(0, 1))
        west = npx.sum(west * (vs.line_dir_west_mask[i, j] & vs.boundary_mask[i, j]), axis=(0, 1))
        north = npx.sum(north * (vs.line_dir_north_mask[i, j] & vs.boundary_mask[i, j]), axis=(0, 1))
        south = npx.sum(south * (vs.line_dir_south_mask[i, j] & vs.boundary_mask[i, j]), axis=(0, 1))
        return global_sum(east + west + north + south)

    elif kind == "full":
        isle_int = npx.empty((nisle, nisle))

        def loop_body(isle, isle_int):
            east_isle = npx.sum(
                east[..., isle, npx.newaxis] * (vs.line_dir_east_mask[i, j] & vs.boundary_mask[i, j]),
                axis=(0, 1),
            )
            west_isle = npx.sum(
                west[..., isle, npx.newaxis] * (vs.line_dir_west_mask[i, j] & vs.boundary_mask[i, j]),
                axis=(0, 1),
            )
            north_isle = npx.sum(
                north[..., isle, npx.newaxis] * (vs.line_dir_north_mask[i, j] & vs.boundary_mask[i, j]),
                axis=(0, 1),
            )
            south_isle = npx.sum(
                south[..., isle, npx.newaxis] * (vs.line_dir_south_mask[i, j] & vs.boundary_mask[i, j]),
                axis=(0, 1),
            )
            isle_int = update(isle_int, at[:, isle], east_isle + west_isle + north_isle + south_isle)
            return isle_int

        isle_int = for_loop(0, nisle, loop_body, isle_int)

        return global_sum(isle_int)

    else:
        raise ValueError('"kind" argument must be "same" or "full"')
