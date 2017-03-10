from netCDF4 import Dataset

from climate.pyom.diagnostics.io_threading import threaded_netcdf
from climate.pyom import pyom_method, variables

@pyom_method
def def_grid_cdf(pyom, ncfile):
    """
    Define standard grid in netcdf file
    """
    if not isinstance(ncfile,Dataset):
        raise TypeError("Argument needs to be a netCDF4 Dataset")

    dims = variables.OUTPUT_DIMENSIONS
    for dim in dims:
        var = pyom.variables[dim]
        nc_dim = ncfile.createDimension(dim, variables.get_dimensions(pyom, var.dims, include_ghosts=False)[0])
        init_var(pyom, dim, var, ncfile)
    nc_dim_time = ncfile.createDimension("Time", None)
    nc_dim_var_time = ncfile.createVariable("Time","f8",("Time",))
    nc_dim_var_time.long_name = "Time"
    nc_dim_var_time.units = "days"
    nc_dim_var_time.time_origin = "01-JAN-1900 00:00:00"


@pyom_method
def panic_snap(pyom):
    print("Writing snapshot before panic shutdown")
    if not pyom.enable_diag_snapshots:
        init_snap_cdf(pyom)
    diag_snap(pyom)


@pyom_method
def init_var(pyom, key, var, ncfile):
    dims = tuple(d for d in var.dims if d in variables.OUTPUT_DIMENSIONS)
    if var.time_dependent:
        dims += ("Time",)
    if not key in ncfile.variables:
        v = ncfile.createVariable(key, var.dtype, dims, fill_value=variables.FILL_VALUE)
        v.long_name = var.name
        v.units = var.units
        v.missing_value = variables.FILL_VALUE
        if not var.time_dependent:
            write_var(pyom, key, var, None, ncfile)


@pyom_method
def init_snap_cdf(pyom):
    """
    initialize NetCDF snapshot file
    """
    print("Preparing file {}".format(pyom.snap_file))
    with threaded_netcdf(pyom, Dataset(pyom.snap_file, "w"), file_id="snapshot") as snap_dataset:
        def_grid_cdf(pyom, snap_dataset)
        for key, var in pyom.variables.items():
            if var.output:
                init_var(pyom,key,var,snap_dataset)

@pyom_method
def write_var(pyom, key, var, n, ncfile, var_data=None):
    if var_data is None:
        var_data = getattr(pyom,key)
    gridmask = variables.get_grid_mask(pyom,var.dims)
    if not gridmask is None:
        newaxes = (slice(None),) * gridmask.ndim + (np.newaxis,) * (var_data.ndim - gridmask.ndim)
        var_data = np.where(gridmask.astype(np.bool)[newaxes], var_data, variables.FILL_VALUE)
    var_data = variables.remove_ghosts(var_data, var.dims) * var.scale
    tmask = tuple(pyom.tau if dim == variables.TIMESTEPS[0] else slice(None) for dim in var.dims)
    if "Time" in ncfile[key].dimensions:
        ncfile[key][..., n] = var_data[tmask]
    else:
        ncfile[key][...] = var_data[tmask]


@pyom_method
def diag_snap(pyom):
    time_in_days = pyom.itt * pyom.dt_tracer / 86400.
    if time_in_days < 1.0:
        print(" writing snapshot at {}s".format(time_in_days * 86400.))
    else:
        print(" writing snapshot at {}d".format(time_in_days))

    with threaded_netcdf(pyom, Dataset(pyom.snap_file, "a"), file_id="snapshot") as snap_dataset:
        snapshot_number = snap_dataset["Time"].size
        snap_dataset["Time"][snapshot_number] = time_in_days
        for key, var in pyom.variables.items():
            if var.output and var.time_dependent:
                write_var(pyom, key, var, snapshot_number, snap_dataset)
