import os

from loguru import logger

from .io_tools import netcdf as nctools, hdf5 as h5tools
from ..decorators import veros_method, do_not_disturb
from .. import time, runtime_state, distributed, runtime_settings


class VerosDiagnostic:
    """Base class for diagnostics. Provides an interface and wrappers for common I/O.

    Any diagnostic needs to implement the five interface methods and set some attributes.
    """
    name = None #: Name that identifies the current diagnostic
    sampling_frequency = 0.
    output_frequency = 0.
    output_path = None

    def __init__(self, vs):
        pass

    def _not_implemented(self, vs):
        raise NotImplementedError('must be implemented by subclass')

    initialize = _not_implemented
    """Called at the end of setup. Use this to process user settings and handle setup."""

    diagnose = _not_implemented
    """Called with frequency ``sampling_frequency``."""

    output = _not_implemented
    """Called with frequency ``output_frequency``."""

    write_restart = _not_implemented
    """Responsible for writing restart files."""

    read_restart = _not_implemented
    """Responsible for reading restart files."""

    @veros_method
    def get_output_file_name(self, vs):
        return self.output_path.format(**vars(vs))

    @do_not_disturb
    @veros_method
    def initialize_output(self, vs, variables, var_data=None, extra_dimensions=None):
        if vs.diskless_mode or (not self.output_frequency and not self.sampling_frequency):
            return
        output_path = self.get_output_file_name(vs)
        if os.path.isfile(output_path) and not vs.force_overwrite:
            raise IOError('output file {} for diagnostic "{}" exists '
                          '(change output path or enable force_overwrite setting)'
                          .format(output_path, self.name))
        with nctools.threaded_io(vs, output_path, 'w') as outfile:
            nctools.initialize_file(vs, outfile)
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    nctools.add_dimension(vs, dim_id, size, outfile)
            for key, var in variables.items():
                if key not in outfile.variables:
                    nctools.initialize_variable(vs, key, var, outfile)
                if not var.time_dependent:
                    if var_data is None or key not in var_data:
                        raise ValueError('var_data argument must be given for constant variables')
                    nctools.write_variable(vs, key, var, var_data[key], outfile)

    @do_not_disturb
    @veros_method
    def write_output(self, vs, variables, variable_data):
        if vs.diskless_mode:
            return
        with nctools.threaded_io(vs, self.get_output_file_name(vs), 'r+') as outfile:
            time_step = nctools.get_current_timestep(vs, outfile)
            current_days = time.convert_time(vs.time, 'seconds', 'days')
            nctools.advance_time(vs, time_step, current_days, outfile)
            for key, var in variables.items():
                nctools.write_variable(vs, key, var, variable_data[key],
                                       outfile, time_step=time_step)

    @veros_method
    def read_h5_restart(self, vs, var_meta, restart_filename):
        if not os.path.isfile(restart_filename):
            raise IOError('restart file {} not found'.format(restart_filename))

        logger.info(' Reading restart data for diagnostic {} from {}',
                    self.name, restart_filename)

        with h5tools.threaded_io(vs, restart_filename, 'r') as infile:
            variables = {}
            for key, var in infile[self.name].items():
                if np.isscalar(var):
                    variables[key] = var
                    continue

                local_shape = distributed.get_local_size(vs, var.shape, var_meta[key].dims, include_overlap=True)
                gidx, lidx = distributed.get_chunk_slices(vs, var_meta[key].dims[:var.ndim], include_overlap=True)

                variables[key] = np.empty(local_shape, dtype=str(var.dtype))

                if runtime_settings.backend == 'bohrium':
                    variables[key][lidx] = var[gidx].astype(variables[key].dtype)
                else:
                    variables[key][lidx] = var[gidx]

                distributed.exchange_overlap(vs, variables[key], var_meta[key].dims)

            attributes = {key: var.item() for key, var in infile[self.name].attrs.items()}

        return attributes, variables

    @do_not_disturb
    @veros_method
    def write_h5_restart(self, vs, attributes, var_meta, var_data, outfile):
        group = outfile.require_group(self.name)
        for key, var in var_data.items():
            try:
                var = var.copy2numpy()
            except AttributeError:
                pass

            global_shape = distributed.get_global_size(vs, var.shape, var_meta[key].dims, include_overlap=True)
            gidx, lidx = distributed.get_chunk_slices(vs, var_meta[key].dims, include_overlap=True)

            kwargs = dict(
                exact=True,
                chunks=tuple(
                    distributed.get_local_size(vs, var.shape, var_meta[key].dims, include_overlap=False)
                )
            )
            if vs.enable_hdf5_gzip_compression and runtime_state.proc_num == 1:
                kwargs.update(
                    compression='gzip',
                    compression_opts=9
                )
            group.require_dataset(key, global_shape, var.dtype, **kwargs)
            group[key][gidx] = var[lidx]

        for key, val in attributes.items():
            try:
                val = val.copy2numpy()
            except AttributeError:
                pass

            group.attrs[key] = val
