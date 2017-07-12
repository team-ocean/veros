import numpy as np
import scipy.interpolate
import scipy.spatial


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
                                    bounds_error=False, fill_value=np.nan, method=kind)

    if fill:
        var = fill_holes(var)
    return var


def fill_holes(data):
    """A simple inpainting function that replaces NaN values in `data` with the
    nearest finite value.
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


def make_cyclic(longitude, array=None, wrap=360.):
    """Create a cyclic version of a longitude array and (optionally) another array.

    Arguments:
        longitude (ndarray): Longitude array of shape (nlon, ...).
        array (ndarray): Another array that is to be made cyclic of shape (nlon, ...).
        wrap (float): Wrapping value, defaults to 360 (degrees).

    Returns:
        Tuple containing (cyclic_longitudes, cyclic_array) if `array` is given, otherwise
        just the ndarray cyclic_longitudes of shape (2 * nlon, ...).
    """
    lonsize = longitude.shape[0]
    cyclic_longitudes = np.hstack((longitude[lonsize//2:, ...] - wrap, longitude, longitude[:lonsize//2, ...] + wrap))
    if array is None:
        return cyclic_longitudes
    cyclic_array = np.hstack((array[lonsize//2:, ...], array, array[:lonsize//2, ...]))
    return cyclic_longitudes, cyclic_array


def get_coastline_distance(coords, coast_mask, spherical=False, radius=None, num_candidates=None, n_jobs=-1):
    """Calculate the (approximate) distance of each water cell from the nearest coastline.

    Arguments:
        coords (tuple of ndarrays): Tuple containing x and y (longitude and latitude)
            coordinate arrays of shape (nx, ny).
        coast_mask (ndarray): Boolean mask indicating whether a cell is a land cell
            (must be same shape as coordinate arrays).
        spherical (bool): Use spherical instead of Cartesian coordinates.
            When this is `True`, cyclical boundary conditions are used, and the
            resulting distances are only approximate. Cells are pre-sorted by
            Euclidean lon-lat distance, and great circle distances are calculated for
            the first `num_candidates` elements. Defaults to `False`.
        radius (float): Radius of spherical coordinate system. Must be given when
            `spherical` is `True`.
        num_candidates (int): Number of candidates to calculate great circle distances
            for for each water cell. The higher this value, the more accurate the returned
            distances become when `spherical` is `True`. Defaults to the square root
            of the number of coastal cells.
        n_jobs (int): Number of parallel jobs to determine nearest neighbors
            (defaults to -1, which uses all available threads).

    Returns:
        :obj:`ndarray` of shape (nx, ny) indicating the distance to the nearest land
        cell (0 if cell is land).

    Example:
        The following returns coastal distances of all T cells for a spherical Veros setup.

        >>> coords = np.meshgrid(self.xt[2:-2], self.yt[2:-2], indexing="ij")
        >>> dist = tools.get_coastline_distance(coords, self.kbot > 0, spherical=True, radius=self.radius)
    """
    if not len(coords) == 2:
        raise ValueError("coords must be lon-lat tuple")
    if not all(c.shape == coast_mask.shape for c in coords):
        raise ValueError("coordinates must have same shape as coastal mask")
    if spherical and not radius:
        raise ValueError("radius must be given for spherical coordinates")

    watercoords = np.array([c[~coast_mask] for c in coords]).T
    if spherical:
        coastcoords = np.array(make_cyclic(coords[0][coast_mask], coords[1][coast_mask])).T
    else:
        coastcoords = np.array((coords[0][coast_mask], coords[1][coast_mask])).T
    coast_kdtree = scipy.spatial.cKDTree(coastcoords)

    distance = np.zeros(coords[0].shape)
    if spherical:
        def spherical_distance(coords1, coords2):
            """Calculate great circle distance from latitude and longitude"""
            coords1 *= np.pi / 180.
            coords2 *= np.pi / 180.
            lon1, lon2, lat1, lat2 = coords1[..., 0], coords2[..., 0], coords1[..., 1], coords2[..., 1]
            return radius * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
        if not num_candidates:
            num_candidates = int(np.sqrt(np.count_nonzero(~coast_mask)))
        i_nearest = coast_kdtree.query(watercoords, k=num_candidates, n_jobs=n_jobs)[1]
        approx_nearest = coastcoords[i_nearest]
        distance[~coast_mask] = np.min(spherical_distance(approx_nearest, watercoords[..., np.newaxis, :]), axis=-1)
    else:
        distance[~coast_mask] = coast_kdtree.query(watercoords, n_jobs=n_jobs)[0]

    return distance
