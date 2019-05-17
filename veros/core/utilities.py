from .. import veros_method, runtime_settings as rs, runtime_state as rst


@veros_method
def enforce_boundaries(vs, arr, local=False):
    from ..distributed import exchange_cyclic_boundaries, exchange_overlap
    from ..decorators import CONTEXT

    if vs.enable_cyclic_x:
        if rs.num_proc[0] == 1 or not CONTEXT.is_dist_safe or local:
            arr[-2:, ...] = arr[2:4, ...]
            arr[:2, ...] = arr[-4:-2, ...]
        else:
            exchange_cyclic_boundaries(vs, arr)

    if local or rst.proc_num == 1:
        return

    exchange_overlap(vs, arr, ['xt', 'yt'])


@veros_method(inline=True)
def where(vs, cond, arr1, arr2):
    assert cond.dtype == np.bool
    return cond * arr1 + np.logical_not(cond) * arr2


@veros_method(inline=True)
def pad_z_edges(vs, array):
    """
    Pads the z-axis of an array by repeating its edge values
    """
    if array.ndim == 1:
        newarray = np.zeros(array.shape[0] + 2, dtype=array.dtype)
        newarray[1:-1] = array
        newarray[0] = array[0]
        newarray[-1] = array[-1]
    elif array.ndim >= 3:
        a = list(array.shape)
        a[2] += 2
        newarray = np.zeros(a, dtype=array.dtype)
        newarray[:, :, 1:-1, ...] = array
        newarray[:, :, 0, ...] = array[:, :, 0, ...]
        newarray[:, :, -1, ...] = array[:, :, -1, ...]
    else:
        raise ValueError('Array to pad needs to have 1 or at least 3 dimensions')
    return newarray


@veros_method(inline=True)
def solve_implicit(vs, ks, a, b, c, d, b_edge=None, d_edge=None):
    from .numerics import solve_tridiag  # avoid circular import

    land_mask = (ks >= 0)[:, :, np.newaxis]
    edge_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
                             == ks[:, :, np.newaxis])
    water_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
                              >= ks[:, :, np.newaxis])

    a_tri = water_mask * a * np.logical_not(edge_mask)
    #a_tri = np.logical_not(edge_mask) * a_tri
    b_tri = where(vs, water_mask, b, 1.)
    if b_edge is not None:
        b_tri = where(vs, edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(vs, edge_mask, d_edge, d_tri)

    return solve_tridiag(vs, a_tri, b_tri, c_tri, d_tri), water_mask
