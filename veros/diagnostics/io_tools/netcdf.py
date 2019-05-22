import threading
import contextlib

from loguru import logger

from ... import veros_method, variables, runtime_state, runtime_settings as rs, distributed

"""
netCDF output is designed to follow the COARDS guidelines from
http://ferret.pmel.noaa.gov/Ferret/documentation/coards-netcdf-conventions
"""


@veros_method
def initialize_file(vs, ncfile, create_time_dimension=True):
    """
    Define standard grid in netcdf file
    """
    import h5netcdf

    if not isinstance(ncfile, h5netcdf.File):
        raise TypeError('Argument needs to be a netCDF4 Dataset')

    for dim in variables.BASE_DIMENSIONS:
        var = vs.variables[dim]
        dimsize = variables.get_dimensions(vs, var.dims[::-1], include_ghosts=False, local=False)[0]
        add_dimension(vs, dim, dimsize, ncfile)
        initialize_variable(vs, dim, var, ncfile)
        write_variable(vs, dim, var, getattr(vs, dim), ncfile)

    if create_time_dimension:
        ncfile.dimensions['Time'] = None
        nc_dim_var_time = ncfile.create_variable('Time', ('Time',), float)
        nc_dim_var_time.long_name = 'Time'
        nc_dim_var_time.units = 'days'
        nc_dim_var_time.time_origin = '01-JAN-1900 00:00:00'


@veros_method
def add_dimension(vs, identifier, size, ncfile):
    ncfile.dimensions[identifier] = size


@veros_method
def initialize_variable(vs, key, var, ncfile):
    dims = tuple(d for d in var.dims if d in ncfile.dimensions)
    if var.time_dependent and 'Time' in ncfile.dimensions:
        dims += ('Time',)

    if key in ncfile.variables:
        logger.warning('Variable {} already initialized'.format(key))
        return

    kwargs = {}
    if vs.enable_hdf5_gzip_compression and runtime_state.proc_num == 1:
        kwargs.update(
            compression='gzip',
            compression_opts=9
        )

    global_shape = [ncfile.dimensions[dim] or 1 for dim in dims]
    chunksize = distributed.get_local_size(vs, global_shape, dims, include_overlap=False)

    # transpose all dimensions in netCDF output (convention in most ocean models)
    v = ncfile.create_variable(
        key, dims[::-1], var.dtype or vs.default_float_type,
        fillvalue=variables.FILL_VALUE,
        chunks=tuple(chunksize[::-1]),
        **kwargs
    )
    v.long_name = var.name
    v.units = var.units
    v.missing_value = variables.FILL_VALUE
    v.attrs.update(var.extra_attributes)


@veros_method
def get_current_timestep(vs, ncfile):
    return len(ncfile.variables['Time'])


@veros_method
def advance_time(vs, time_step, time_value, ncfile):
    ncfile.resize_dimension('Time', time_step + 1)
    ncfile.variables['Time'][time_step] = time_value


@veros_method
def write_variable(vs, key, var, var_data, ncfile, time_step=None):
    var_data = var_data * var.scale

    gridmask = variables.get_grid_mask(vs, var.dims)
    if gridmask is not None:
        newaxes = (slice(None),) * gridmask.ndim + (np.newaxis,) * (var_data.ndim - gridmask.ndim)
        var_data = np.where(gridmask.astype(np.bool)[newaxes], var_data, variables.FILL_VALUE)

    if not np.isscalar(var_data):
        tmask = tuple(vs.tau if dim in variables.TIMESTEPS else slice(None) for dim in var.dims)
        var_data = variables.remove_ghosts(var_data, var.dims)[tmask].T

    try:
        var_data = var_data.copy2numpy()
    except AttributeError:
        pass

    var_obj = ncfile.variables[key]

    chunk, _ = distributed.get_chunk_slices(vs, var_obj.dimensions)

    if 'Time' in var_obj.dimensions:
        assert var_obj.dimensions[0] == 'Time'

        if time_step is None:
            raise ValueError('time step must be given for non-constant data')

        chunk = (time_step,) + chunk[1:]

    var_obj[chunk] = var_data


@veros_method
@contextlib.contextmanager
def threaded_io(vs, filepath, mode):
    """
    If using IO threads, start a new thread to write the netCDF data to disk.
    """
    import h5netcdf

    if vs.use_io_threads:
        _wait_for_disk(vs, filepath)
        _io_locks[filepath].clear()

    kwargs = {}
    if runtime_state.proc_num > 1:
        kwargs.update(
            driver='mpio',
            comm=rs.mpi_comm
        )

    nc_dataset = h5netcdf.File(filepath, mode, **kwargs)
    try:
        yield nc_dataset
    finally:
        if vs.use_io_threads:
            threading.Thread(target=_write_to_disk, args=(vs, nc_dataset, filepath)).start()
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
    logger.debug('Waiting for lock {} to be released'.format(file_id))
    _add_to_locks(file_id)
    lock_released = _io_locks[file_id].wait(vs.io_timeout)
    if not lock_released:
        raise RuntimeError('Timeout while waiting for disk IO to finish')


def _write_to_disk(vs, ncfile, file_id):
    """
    Sync netCDF data to disk, close file handle, and release lock.
    May run in a separate thread.
    """
    try:
        ncfile.close()
    finally:
        if vs.use_io_threads and file_id is not None:
            _io_locks[file_id].set()
