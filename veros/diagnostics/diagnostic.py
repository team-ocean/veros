import os

from veros import logger

from veros.core.operators import numpy as np
from veros.diagnostics.io_tools import netcdf as nctools, hdf5 as h5tools
from veros.signals import do_not_disturb
from veros import time, runtime_state, distributed, variables, runtime_settings


class VerosDiagnostic:
    """Base class for diagnostics. Provides an interface and wrappers for common I/O.

    Any diagnostic needs to implement the five interface methods and set some attributes.
    """
    name = None #: Name that identifies the current diagnostic
    sampling_frequency = 0.
    output_frequency = 0.
    output_path = None

    def _not_implemented(self, state):
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

    def get_output_file_name(self, state):
        statedict = dict(state.variables.items())
        statedict.update(state.settings.items())
        return self.output_path.format(**statedict)

    @do_not_disturb
    def initialize_output(self, state, variables, var_data=None, extra_dimensions=None):
        settings = state.settings

        if settings.diskless_mode or (not self.output_frequency and not self.sampling_frequency):
            return

        output_path = self.get_output_file_name(state)
        if os.path.isfile(output_path) and not settings.force_overwrite:
            raise IOError('output file {} for diagnostic "{}" exists '
                          '(change output path or enable force_overwrite setting)'
                          .format(output_path, self.name))

        # possible race condition ahead!
        distributed.barrier()

        with nctools.threaded_io(state, output_path, 'w') as outfile:
            nctools.initialize_file(state, outfile)
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    nctools.add_dimension(state, dim_id, size, outfile)

            for key, var in variables.items():
                if key not in outfile.variables:
                    nctools.initialize_variable(state, key, var, outfile)

                if not var.time_dependent:
                    if var_data is None or key not in var_data:
                        raise ValueError('var_data argument must be given for constant variables')

                    nctools.write_variable(state, key, var, var_data[key], outfile)

    @do_not_disturb
    def write_output(self, state, variables, variable_data):
        vs = state.variables
        settings = state.settings

        if settings.diskless_mode:
            return

        with nctools.threaded_io(state, self.get_output_file_name(state), 'r+') as outfile:
            time_step = nctools.get_current_timestep(state, outfile)
            current_days = time.convert_time(vs.time, 'seconds', 'days')
            nctools.advance_time(state, time_step, current_days, outfile)
            for key, var in variables.items():
                nctools.write_variable(state, key, var, variable_data[key],
                                       outfile, time_step=time_step)

    def read_h5_restart(self, state, var_meta, restart_filename):
        if not os.path.isfile(restart_filename):
            raise IOError('restart file {} not found'.format(restart_filename))

        logger.info(' Reading restart data for diagnostic {} from {}',
                    self.name, restart_filename)

        with h5tools.threaded_io(state, restart_filename, 'r') as infile:
            variables = {}
            for key, var in infile[self.name].items():
                if np.isscalar(var):
                    variables[key] = var
                    continue

                local_shape = distributed.get_local_size(state, var.shape, var_meta[key].dims, include_overlap=True)
                gidx, lidx = distributed.get_chunk_slices(state, var_meta[key].dims[:var.ndim], include_overlap=True)

                variables[key] = np.empty(local_shape, dtype=str(var.dtype))
                variables[key][lidx] = var[gidx]

                distributed.exchange_overlap(state, variables[key], var_meta[key].dims)

            attributes = {key: var.item() for key, var in infile[self.name].attrs.items()}

        return attributes, variables

    @do_not_disturb
    def write_h5_restart(self, state, attributes, var_meta, var_data, outfile):
        settings = state.settings

        group = outfile.require_group(self.name)
        for key, var in var_data.items():
            global_shape = variables.get_shape(state.dimensions, var.shape, var_meta[key].dims, local=False)
            gidx, lidx = distributed.get_chunk_slices(settings.nx, settings.ny, var_meta[key].dims, include_overlap=True)

            chunksize = tuple(
                variables.get_shape(state.dimensions, (d,), local=True, include_ghosts=False)[0]
                if d in state.dimensions else 1
                for d in var_meta[key].dims
            )

            kwargs = dict(
                exact=True,
                chunks=chunksize,
            )
            if runtime_settings.hdf5_gzip_compression and runtime_state.proc_num == 1:
                kwargs.update(
                    compression='gzip',
                    compression_opts=1
                )
            group.require_dataset(key, global_shape, var.dtype, **kwargs)
            group[key][gidx] = var[lidx]

        for key, val in attributes.items():
            group.attrs[key] = val
