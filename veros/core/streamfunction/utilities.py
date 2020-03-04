import numpy as np

from veros.core import veros_kernel
from veros.distributed import global_sum


@veros_kernel(static_args=('kind',))
def line_integrals(dxu, dyu, cost, line_dir_east_mask, line_dir_west_mask,
                   line_dir_north_mask, line_dir_south_mask, boundary_mask,
                   nisle, uloc, vloc, kind='same'):
    """
    calculate line integrals along all islands

    Arguments:
        kind: 'same' calculates only line integral contributions of an island with itself,
               while 'full' calculates all possible pairings between all islands.
    """
    if kind not in ('same', 'full'):
        raise ValueError('kind must be "same" or "full"')

    east = vloc[1:-2, 1:-2, :] * dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * dxu[1:-2, np.newaxis, np.newaxis] \
        * cost[np.newaxis, 2:-1, np.newaxis]
    west = -vloc[2:-1, 1:-2, :] * dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * dxu[1:-2, np.newaxis, np.newaxis] \
        * cost[np.newaxis, 1:-2, np.newaxis]
    north = vloc[1:-2, 1:-2, :] * dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * dxu[1:-2, np.newaxis, np.newaxis] \
        * cost[np.newaxis, 1:-2, np.newaxis]
    south = -vloc[2:-1, 1:-2, :] * dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * dxu[1:-2, np.newaxis, np.newaxis] \
        * cost[np.newaxis, 2:-1, np.newaxis]

    if kind == 'same':
        east = np.sum(east * (line_dir_east_mask[1:-2, 1:-2] &
                              boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        west = np.sum(west * (line_dir_west_mask[1:-2, 1:-2] &
                              boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        north = np.sum(north * (line_dir_north_mask[1:-2, 1:-2]
                                & boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        south = np.sum(south * (line_dir_south_mask[1:-2, 1:-2]
                                & boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        out = east + west + north + south
    else:
        out = np.empty((nisle, nisle))
        for isle in range(nisle):
            east_isle = np.sum(
                east[..., isle, np.newaxis]
                * (line_dir_east_mask[1:-2, 1:-2] & boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            west_isle = np.sum(
                west[..., isle, np.newaxis]
                * (line_dir_west_mask[1:-2, 1:-2] & boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            north_isle = np.sum(
                north[..., isle, np.newaxis]
                * (line_dir_north_mask[1:-2, 1:-2] & boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            south_isle = np.sum(
                south[..., isle, np.newaxis]
                * (line_dir_south_mask[1:-2, 1:-2] & boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            out[:, isle] = east_isle + west_isle + north_isle + south_isle

    return global_sum(out)
