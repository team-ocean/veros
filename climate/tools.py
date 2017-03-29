import numpy as np

def _gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def _normalize_sum(var, sum_value, minimum_value=0.):
    var[:2] = 0.
    var *= (sum_value - len(var) * minimum_value) / var.sum()
    return var + minimum_value

def gaussian_spacing(n, sum_value, min_spacing = 0., mu = 0.5, sigma = 0.125):
    ddx = _gaussian(np.arange(n), mu * n, sigma * n)
    return _normalize_sum(np.cumsum(ddx[::-1]), sum_value, min_spacing)[::-1]

def interpolate(coords, var, interp_coords, missing_value=None, fill=True, kind="linear"):
    if len(coords) != len(interp_coords) or len(coords) != var.ndim:
        raise ValueError("Dimensions of coordinates and values do not match")
    var = np.array(var)
    if not missing_value is None:
        invalid_mask = var == missing_value
        var[invalid_mask] = np.nan
    for i, (x, x_new) in enumerate(zip(coords, interp_coords)):
        var = interpolate_along_axis(x, var, x_new, i, kind=kind)
    if fill:
        var = fill_holes(var)
    return var

def interpolate_along_axis(coords, arr, interp_coords, axis, kind="linear"):
    """Lightweight interpolation along a chosen axis for data on a regular grid.

    """
    if len(coords) != arr.shape[axis]:
        raise ValueError("Length of coordinate array does not match input array shape along chosen axis")

    diff = coords[np.newaxis, :] - interp_coords[:, np.newaxis]

    broadcast_shape = [np.newaxis] * arr.ndim
    broadcast_shape[axis] = slice(None)

    out_shape = list(arr.shape)
    out_shape[axis] = len(interp_coords)

    if kind == "nearest":
        i = np.argmin(np.abs(diff), axis=1)
        full_shape = [np.newaxis] * arr.ndim
        full_shape[axis] = slice(None)
        i_full = i[broadcast_shape] * np.ones(out_shape)
        indices = np.indices(i_full.shape)
        indices[axis] = i_full
        return arr[tuple(indices)]
    elif kind == "linear":
        if not np.all(np.sort(coords) == coords):
            raise ValueError("Coordinates must be strictly ascending for linear interpolation")

        diff_m = np.where(diff <= 0, diff, np.inf)
        i_m = np.argmin(np.abs(diff_m), axis=1)
        i_p = np.minimum(len(coords) - 1, i_m + 1)
        i_m_full = i_m[broadcast_shape] * np.ones(out_shape)
        i_p_full = i_p[broadcast_shape] * np.ones(out_shape)

        pos = np.where(i_p_full == i_m_full, 1., ((coords[i_p] - interp_coords) \
                        / (coords[i_p] - coords[i_m] + 1e-20))[broadcast_shape])

        indices_p, indices_m = np.indices(i_m_full.shape), np.indices(i_m_full.shape)
        indices_p[axis] = i_p_full
        indices_m[axis] = i_m_full
        return arr[tuple(indices_p)] * (1-pos) + arr[tuple(indices_m)] * pos
    else:
        raise ValueError("'kind' must be 'nearest' or 'linear'")


def fill_holes(data):
    data = data.copy()
    shape = data.shape
    dim = data.ndim
    flag = np.zeros(shape, dtype=bool)
    t_ct = int(data.size/5)
    flag[~np.isnan(data)] = True

    slcs = [slice(None)] * dim

    while np.any(~flag):
        for i in range(dim):
            slcs1 = slcs[:]
            slcs2 = slcs[:]
            slcs1[i] = slice(0, -1)
            slcs2[i] = slice(1, None)

            # replace from the right
            repmask = np.logical_and(~flag[slcs1], flag[slcs2])
            data[slcs1][repmask] = data[slcs2][repmask]
            flag[slcs1][repmask] = True

            # replace from the left
            repmask = np.logical_and(~flag[slcs2], flag[slcs1])
            data[slcs2][repmask] = data[slcs1][repmask]
            flag[slcs2][repmask] = True
    return data


def get_periodic_interval(currentTime, cycleLength, recSpacing, nbRec):
    """  interpolation routine taken from mitgcm
    """
    locTime = currentTime - recSpacing * 0.5 + cycleLength * (2 - round(currentTime/cycleLength))
    tmpTime = locTime % cycleLength
    tRec1 = 1 + int(tmpTime/recSpacing)
    tRec2 = 1 + tRec1 % int(nbRec)
    wght2 = (tmpTime - recSpacing*(tRec1 - 1)) / recSpacing
    wght1 = 1.0 - wght2
    return (tRec1-1, wght1), (tRec2-1, wght2)
