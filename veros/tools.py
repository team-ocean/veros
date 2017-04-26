import numpy as np
import scipy.interpolate


def _gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def _normalize_sum(var, sum_value, minimum_value=0.):
    var *= (sum_value - len(var) * minimum_value) / var.sum()
    return var + minimum_value


def gaussian_spacing(n, sum_value, min_spacing=0., mu=0.5, sigma=0.125):
    """Create a sample where values are separated by gaussian step sizes.

    This can be used to create a vertical grid that covers a fixed distance
    (``sum_value``) and minimizes discretization errors.

    .. note::
        The first two step sizes are kept constant at ``minimum_value``.

    Arguments:
       n: Number of sample members.
       sum_value: Target sum of the created sample.
       min_spacing (optional): Minimum spacing between sample members. Defaults
          to zero.
       mu (optional): Mean of the underlying gaussian.
       sigma (optional): Standard deviation of the underlying gaussian.

    Example:
       >>> a = gaussian_spacing(10, 100., min_spacing=2.)
       >>> a
       array([  2.        ,   2.        ,   2.10115551,   2.65269846,
         4.38051667,   7.72821622,  12.25219105,  17.2265533 ,
        22.29184721,  27.36682157])
       >>> a.sum()
       100.00000000000001

    """
    ddx = _gaussian(np.arange(n), mu * n, sigma * n)
    dx = np.cumsum(ddx)
    dx[:2] = 0.
    return _normalize_sum(np.cumsum(dx), sum_value, min_spacing)


def interpolate(coords, var, interp_coords, missing_value=None, fill=True, kind="linear"):
    """Interpolate globally defined data to a different (regular) grid.

    Arguments:
       coords: Tuple of coordinate arrays for each dimension.
       var (:obj:`ndarray` of dim (nx1, ..., nxd)): Variable data to interpolate.
       interp_coords: Tuple of coordinate arrays to interpolate to.
       missing_value (optional): Value denoting cells of missing data in ``var``.
          Is replaced by `NaN` before interpolating. Defaults to `None`, which means
          no replacement is taking place.
       fill (bool, optional): Whether `NaN` values should be replaced by the nearest
          finite value after interpolating. Defaults to ``True``.
       kind (str, optional): Order of interpolation. Supported are `nearest` and
          `linear` (default).

    Returns:
       :obj:`ndarray` containing the interpolated values on the grid spanned by
       ``interp_coords``.
    """
    if len(coords) != len(interp_coords) or len(coords) != var.ndim:
        raise ValueError("Dimensions of coordinates and values do not match")
    var = np.array(var)
    if missing_value is not None:
        invalid_mask = np.isclose(var, missing_value)
        var[invalid_mask] = np.nan
    if var.ndim > 1 and coords[0].ndim == 1:
        interp_grid = np.rollaxis(np.array(np.meshgrid(
            *interp_coords, indexing="ij", copy=False)), 0, len(interp_coords) + 1)
    else:
        interp_grid = coords
    var = scipy.interpolate.interpn(coords, var, interp_grid,
                                    bounds_error=False, fill_value=None, method=kind)

    if fill:
        var = fill_holes(var)
    return var


def fill_holes(data):
    """A simple helper function that replaces NaN values with the nearest finite value.
    """
    data = data.copy()
    shape = data.shape
    dim = data.ndim
    flag = np.zeros(shape, dtype=bool)
    t_ct = int(data.size / 5)
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


def get_periodic_interval(current_time, cycle_length, rec_spacing, n_rec):
    """Used for linear interpolation between periodic time intervals.

    One common application is the interpolation of external forcings that are defined
    at discrete times (e.g. one value per month of a standard year) to the current
    time step.

    Arguments:
       current_time (float): Time to interpolate to.
       cycle_length (float): Total length of one periodic cycle.
       rec_spacing (float): Time spacing between each data record.
       n_rec (int): Total number of records available.

    Returns:
       :obj:`tuple` containing (n1, f1), (n2, f2): Indices and weights for the interpolated
       record array.

    Example:
       The following interpolates a record array ``data`` containing 12 monthly values
       to the current time step:

       >>> year_in_seconds = 60. * 60. * 24. * 365.
       >>> current_time = 60. * 60. * 24. * 45. # mid-february
       >>> print(data.shape)
       (360, 180, 12)
       >>> (n1, f1), (n2, f2) = get_periodic_interval(current_time, year_in_seconds, year_in_seconds / 12, 12)
       >>> data_at_current_time = f1 * data[..., n1] + f2 * data[..., n2]
    """
    locTime = current_time - rec_spacing * 0.5 + \
        cycle_length * (2 - round(current_time / cycle_length))
    tmpTime = locTime % cycle_length
    tRec1 = 1 + int(tmpTime / rec_spacing)
    tRec2 = 1 + tRec1 % int(n_rec)
    wght2 = (tmpTime - rec_spacing * (tRec1 - 1)) / rec_spacing
    wght1 = 1.0 - wght2
    return (tRec1 - 1, wght1), (tRec2 - 1, wght2)
