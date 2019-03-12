import threading
import contextlib
import logging
import warnings

from ... import veros_method, variables

"""
netCDF output is designed to follow the COARDS guidelines from
http://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions
"""


@veros_method
def initialize_file(vs, ncfile, create_time_dimension=True):
    """
    Define standard grid in netcdf file
    """
    from netCDF4 import Dataset
    if not isinstance(ncfile, Dataset):
        raise TypeError("Argument needs to be a netCDF4 Dataset")

    for dim in variables.BASE_DIMENSIONS:
        var = vs.variables[dim]
        dimsize = variables.get_dimensions(vs, var.dims[::-1], include_ghosts=False)[0]
        nc_dim = add_dimension(vs, dim, dimsize, ncfile)
        initialize_variable(vs, dim, var, ncfile)
        write_variable(vs, dim, var, getattr(vs, dim), ncfile)

    if create_time_dimension:
        nc_dim_time = ncfile.createDimension("Time", None)
        nc_dim_var_time = ncfile.createVariable("Time", "f8", ("Time",))
        nc_dim_var_time.long_name = "Time"
        nc_dim_var_time.units = "days"
        nc_dim_var_time.time_origin = "01-JAN-1900 00:00:00"


@veros_method
def add_dimension(vs, identifier, size, ncfile):
    return ncfile.createDimension(identifier, size)


@veros_method
def initialize_variable(vs, key, var, ncfile):
    dims = tuple(d for d in var.dims if d in ncfile.dimensions)
    if var.time_dependent and "Time" in ncfile.dimensions:
        dims += ("Time",)
    if key in ncfile.variables:
        warnings.warn("Variable {} already initialized".format(key))
        return
    # transpose all dimensions in netCDF output (convention in most ocean models)
    v = ncfile.createVariable(key, var.dtype or vs.default_float_type, dims[::-1],
                              fill_value=variables.FILL_VALUE,
                              zlib=vs.enable_netcdf_zlib_compression)
    v.long_name = var.name
    v.units = var.units
    v.missing_value = variables.FILL_VALUE
    for extra_key, extra_attr in var.extra_attributes.items():
        setattr(v, extra_key, extra_attr)


@veros_method
def get_current_timestep(vs, ncfile):
    return len(ncfile.dimensions["Time"])


@veros_method
def advance_time(vs, time_step, time_value, ncfile):
    ncfile.variables["Time"][time_step] = time_value


@veros_method
def write_variable(vs, key, var, var_data, ncfile, time_step=None):
    gridmask = variables.get_grid_mask(vs, var.dims)
    if gridmask is not None:
        newaxes = (slice(None),) * gridmask.ndim + (np.newaxis,) * (var_data.ndim - gridmask.ndim)
        var_data = np.where(gridmask.astype(np.bool)[newaxes], var_data, variables.FILL_VALUE)
    if not np.isscalar(var_data):
        tmask = tuple(vs.tau if dim in variables.TIMESTEPS else slice(None) for dim in var.dims)
        var_data = variables.remove_ghosts(var_data, var.dims)[tmask].T
    var_data = var_data * var.scale
    if "Time" in ncfile.variables[key].dimensions:
        if time_step is None:
            raise ValueError("time step must be given for non-constant data")
        try:
            ncfile.variables[key][time_step, ...] = var_data.copy2numpy()
        except AttributeError:
            ncfile.variables[key][time_step, ...] = var_data
    else:
        try:
            ncfile.variables[key][...] = var_data.copy2numpy()
        except AttributeError:
            ncfile.variables[key][...] = var_data


@veros_method
@contextlib.contextmanager
def threaded_io(vs, filepath, mode):
    """
    If using IO threads, start a new thread to write the netCDF data to disk.
    """
    from netCDF4 import Dataset
    if vs.use_io_threads:
        _wait_for_disk(vs, filepath)
        _io_locks[filepath].clear()
    nc_dataset = Dataset(filepath, mode)
    try:
        yield nc_dataset
    finally:
        if vs.use_io_threads:
            io_thread = threading.Thread(target=_write_to_disk, args=(vs, nc_dataset, filepath))
            io_thread.start()
        else:
            _write_to_disk(vs, nc_dataset, filepath)


_io_locks = {}


def _add_to_locks(file_id):
    """
    If there is no lock for file_id, create one
    """
    if file_id not in _io_locks:
        _io_locks[file_id] = threading.Event()
        _io_locks[file_id].set()


def _wait_for_disk(vs, file_id):
    """
    Wait for the lock of file_id to be released
    """
    logging.debug("Waiting for lock {} to be released".format(file_id))
    _add_to_locks(file_id)
    lock_released = _io_locks[file_id].wait(vs.io_timeout)
    if not lock_released:
        raise RuntimeError("Timeout while waiting for disk IO to finish")


def _write_to_disk(vs, ncfile, file_id):
    """
    Sync netCDF data to disk, close file handle, and release lock.
    May run in a separate thread.
    """
    ncfile.sync()
    ncfile.close()
    if vs.use_io_threads and file_id is not None:
        _io_locks[file_id].set()
