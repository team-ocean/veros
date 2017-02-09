import numpy as np

import climate.pyom.numerics

def pad_z_edges(array):
    """
    Pads the z-axis of an array by repeating its edge values
    """
    if array.ndim == 1:
        newarray = np.empty(array.shape[0] + 2)
        newarray[1:-1] = array
        newarray[0] = array[0]
        newarray[-1] = array[-1]
    elif array.ndim >= 3:
        a = list(array.shape)
        a[2] += 2
        newarray = np.empty(a)
        newarray[:,:,1:-1,...] = array
        newarray[:,:,0,...] = array[:,:,0,...]
        newarray[:,:,-1,...] = array[:,:,-1,...]
    else:
        raise ValueError("Array to pad needs to have 1 or at least 3 dimensions")
    return newarray


def solve_implicit(ks, a, b, c, d, pyom, b_edge=None, d_edge=None):
    land_mask = (ks >= 0)[:,:,None]
    if not np.count_nonzero(land_mask):
        return np.zeros_like(land_mask), np.zeros_like(land_mask)

    edge_mask = land_mask & (np.indices((a.shape))[2] == ks[:,:,None])
    water_mask = land_mask & (np.indices((a.shape))[2] >= ks[:,:,None])

    a_tri = np.zeros_like(a)
    b_tri = np.zeros_like(b)
    c_tri = np.zeros_like(c)
    d_tri = np.zeros_like(d)

    a_tri[:,:,1:] = a[:,:,1:]
    a_tri[edge_mask] = 0.
    b_tri[:,:,1:] = b[:,:,1:]
    if not (b_edge is None):
        b_tri[edge_mask] = b_edge[edge_mask]
    c_tri[:,:,:-1] = c[:,:,:-1]
    c_tri[:,:,-1] = 0.
    d_tri[...] = d
    if not (d_edge is None):
        d_tri[edge_mask] = d_edge[edge_mask]
    return climate.pyom.numerics.solve_tridiag(a_tri[water_mask],b_tri[water_mask],c_tri[water_mask],d_tri[water_mask]), water_mask
