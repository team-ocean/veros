import numpy as np

def _gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def _normalize_sum(var, sum_value, minimum_value=0.):
    var[0] = 0.
    var *= (sum_value - len(var) * minimum_value) / var.sum()
    return var + minimum_value

def gaussian_spacing(n, sum_value, min_spacing = 0., mu = 0.5, sigma = 0.125):
    ddx = _gaussian(np.arange(n), mu * n, sigma * n)
    return _normalize_sum(np.cumsum(ddx), sum_value, min_spacing)

def interpolate_along_axis(coords, arr, interp_coords, axis):
    assert len(coords) == arr.shape[axis]

    diff = coords[np.newaxis, :] - interp_coords[:, np.newaxis]
    diff_m = np.where(diff <= 0, diff, np.inf)

    i_m = np.argmin(np.abs(diff_m), axis=1)
    i_p = np.minimum(len(coords) - 1, i_m + 1)

    full_shape = [np.newaxis] * arr.ndim
    full_shape[axis] = slice(None)
    s = [slice(None)] * arr.ndim
    s[axis] = i_m
    mask = np.isnan(arr[s])
    i_m_full = np.where(mask, i_p[full_shape], i_m[full_shape])
    s[axis] = i_p
    mask = np.isnan(arr[s])
    i_p_full = np.where(mask, i_m[full_shape], i_p[full_shape])

    pos = np.where(i_p_full == i_m_full, 1., ((coords[i_p] - interp_coords) \
                    / (coords[i_p] - coords[i_m] + 1e-20))[full_shape])

    indices_p, indices_m = np.indices(i_m_full.shape), np.indices(i_m_full.shape)
    indices_p[axis] = i_p_full
    indices_m[axis] = i_m_full
    return arr[tuple(indices_p)] * (1-pos) + arr[tuple(indices_m)] * pos

def fill_holes(data):
    data = data.copy()
    shape = data.shape
    dim = data.ndim
    flag = np.zeros(shape, dtype=bool)
    t_ct = int(data.size/5)
    flag[~np.isnan(data)] = True

    slcs = [slice(None)] * dim

    while np.any(~flag): # as long as there are any False's in flag
        for i in range(dim): # do each axis
            # make slices to shift view one element along the axis
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
