from veros.core.operators import numpy as np

from veros import veros_kernel
from veros.core.operators import update, at


# TODO: handle kwargs in decorator
@veros_kernel(static_args=('enable_cyclic_x', 'local'))
def enforce_boundaries_kernel(arr, enable_cyclic_x, local):
    from .. import runtime_settings as rs
    from ..distributed import exchange_overlap
    from ..decorators import CONTEXT

    if rs.num_proc[0] == 1 or not CONTEXT.is_dist_safe or local:
        if enable_cyclic_x:
            arr = update(arr, at[-2:, ...], arr[2:4, ...])
            arr = update(arr, at[:2, ...], arr[-4:-2, ...])
        return arr

    exchange_overlap(arr, ['xt', 'yt'], cyclic=enable_cyclic_x)
    return arr


def enforce_boundaries(arr, enable_cyclic_x, local=False):
    return enforce_boundaries_kernel(arr, enable_cyclic_x, local)


@veros_kernel
def where(mask, a, b):
    return np.where(mask, a, b)


@veros_kernel
def pad_z_edges(array):
    """
    Pads the z-axis of an array by repeating its edge values
    """
    if array.ndim == 1:
        newarray = np.zeros(array.shape[0] + 2, dtype=array.dtype)
        newarray = update(newarray, at[1:-1], array)
        newarray = update(newarray, at[0], array[0])
        newarray = update(newarray, at[-1], array[-1])
    elif array.ndim >= 3:
        a = list(array.shape)
        a[2] += 2
        newarray = np.zeros(a, dtype=array.dtype)
        newarray = update(newarray, at[:, :, 1:-1, ...], array)
        newarray = update(newarray, at[:, :, 0, ...], array[:, :, 0, ...])
        newarray = update(newarray, at[:, :, -1, ...], array[:, :, -1, ...])
    else:
        raise ValueError('Array to pad needs to have 1 or at least 3 dimensions')
    return newarray


@veros_kernel
def solve_implicit(ks, a, b, c, d, b_edge=None, d_edge=None):
    from .numerics import solve_tridiag  # avoid circular import

    land_mask = (ks >= 0)[:, :, np.newaxis]
    edge_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
                             == ks[:, :, np.newaxis])
    water_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
                              >= ks[:, :, np.newaxis])

    a_tri = water_mask * a * np.logical_not(edge_mask)
    b_tri = where(water_mask, b, 1.)
    if b_edge is not None:
        b_tri = where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(edge_mask, d_edge, d_tri)

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask
