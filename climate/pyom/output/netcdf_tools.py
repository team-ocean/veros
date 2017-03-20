from netCDF4 import Dataset

from .netcdf_threading import threaded_netcdf
from .. import pyom_method, variables

"""
netCDF output is designed to follow the COARDS guidelines from
http://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions
"""

@pyom_method
def initialize_netcdf_file(pyom, ncfile):
    """
    Define standard grid in netcdf file
    """
    if not isinstance(ncfile, Dataset):
        raise TypeError("Argument needs to be a netCDF4 Dataset")

    dims = variables.OUTPUT_DIMENSIONS
    for dim in dims:
        var = pyom.variables[dim]
        nc_dim = ncfile.createDimension(dim, variables.get_dimensions(pyom, var.dims[::-1], include_ghosts=False)[0])
        initialize_variable(pyom, dim, var, ncfile)
    nc_dim_time = ncfile.createDimension("Time", None)
    nc_dim_var_time = ncfile.createVariable("Time","f8",("Time",))
    nc_dim_var_time.long_name = "Time"
    nc_dim_var_time.units = "days"
    nc_dim_var_time.time_origin = "01-JAN-1900 00:00:00"


@pyom_method
def initialize_variable(pyom, key, var, ncfile):
    dims = tuple(d for d in var.dims if d in variables.OUTPUT_DIMENSIONS)
    if var.time_dependent:
        dims += ("Time",)
    if not key in ncfile.variables:
        # revert all dimensions in netCDF output (convention in most ocean models)
        v = ncfile.createVariable(key, var.dtype, dims[::-1],
                                 fill_value=variables.FILL_VALUE,
                                 zlib=pyom.enable_netcdf_zlib_compression)
        v.long_name = var.name
        v.units = var.units
        v.missing_value = variables.FILL_VALUE
        for extra_key, extra_attr in var.extra_attributes.items():
            setattr(v, extra_key, extra_attr)
        if not var.time_dependent:
            write_variable(pyom, key, var, None, ncfile)


@pyom_method
def write_variable(pyom, key, var, n, ncfile, var_data=None):
    if var_data is None:
        var_data = getattr(pyom,key)
    gridmask = variables.get_grid_mask(pyom,var.dims)
    if not gridmask is None:
        newaxes = (slice(None),) * gridmask.ndim + (np.newaxis,) * (var_data.ndim - gridmask.ndim)
        var_data = np.where(gridmask.astype(np.bool)[newaxes], var_data, variables.FILL_VALUE)
    var_data = variables.remove_ghosts(var_data, var.dims) * var.scale
    tmask = tuple(pyom.tau if dim == variables.TIMESTEPS[0] else slice(None) for dim in var.dims)
    if pyom.backend_name == "bohrium":
        var_data = var_data.copy2numpy()
    if "Time" in ncfile.variables[key].dimensions:
        ncfile.variables[key][n, ...] = var_data[tmask].T
    else:
        ncfile.variables[key][...] = var_data[tmask].T
