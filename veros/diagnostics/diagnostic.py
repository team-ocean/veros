from collections import OrderedDict
import os
import copy
import logging
import warnings

from .io_tools import netcdf as nctools
from .io_tools import hdf5 as h5tools
from ..decorators import veros_class_method, veros_method, do_not_disturb
from ..variables import Variable
from .. import time


class VerosDiagnostic(object):
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
        raise NotImplementedError("must be implemented by subclass")

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

    @veros_class_method
    def get_output_file_name(self, vs):
        return self.output_path.format(**vars(vs))

    @do_not_disturb
    @veros_class_method
    def initialize_output(self, vs, variables, var_data=None, extra_dimensions=None):
        if vs.diskless_mode or (not self.output_frequency and not self.sampling_frequency):
            return
        output_path = self.get_output_file_name(vs)
        if os.path.isfile(output_path) and not vs.force_overwrite:
            raise IOError("output file {} for diagnostic '{}' exists "
                          "(change output path or enable force_overwrite setting)"
                          .format(output_path, self.name))
        with nctools.threaded_io(vs, output_path, "w") as outfile:
            nctools.initialize_file(vs, outfile)
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    nctools.add_dimension(vs, dim_id, size, outfile)
            for key, var in variables.items():
                if key not in outfile.variables:
                    nctools.initialize_variable(vs, key, var, outfile)
                if not var.time_dependent:
                    if var_data is None or key not in var_data:
                        raise ValueError("var_data argument must be given for constant variables")
                    nctools.write_variable(vs, key, var, var_data[key], outfile)

    @do_not_disturb
    @veros_class_method
    def write_output(self, vs, variables, variable_data):
        if vs.diskless_mode:
            return
        with nctools.threaded_io(vs, self.get_output_file_name(vs), "r+") as outfile:
            time_step = nctools.get_current_timestep(vs, outfile)
            current_days = time.convert_time(vs, vs.time, "seconds", "days")
            nctools.advance_time(vs, time_step, current_days, outfile)
            for key, var in variables.items():
                nctools.write_variable(vs, key, var, variable_data[key],
                                       outfile, time_step=time_step)

    @veros_class_method
    def get_restart_input_file_name(self, vs):
        """ Returns the file name for input restart file.
        """
        return vs.restart_input_filename.format(**vars(vs))

    @veros_class_method
    def get_restart_output_file_name(self, vs):
        """ Returns the file name for output restart file.
        """
        return vs.restart_output_filename.format(**vars(vs))

    @veros_class_method
    def read_h5_restart(self, vs):
        restart_filename = self.get_restart_input_file_name(vs)
        if not os.path.isfile(restart_filename):
            raise IOError("restart file {} not found".format(restart_filename))

        logging.info(" reading restart data for diagnostic {} from {}"
                     .format(self.name, restart_filename))
        with h5tools.threaded_io(vs, restart_filename, "r") as infile:
            variables = {key: np.array(var[...])
                         for key, var in infile[self.name].items()}
            attributes = {key: var for key, var in infile[self.name].attrs.items()}
        return attributes, variables

    @do_not_disturb
    @veros_class_method
    def write_h5_restart(self, vs, attributes, var_meta, var_data, outfile):
        group = outfile.require_group(self.name)
        for key, var in var_data.items():
            if vs.backend_name == "bohrium" and not np.isscalar(var):
                var = var.copy2numpy()
            kwargs = {"compression": "gzip", "compression_opts": 9} if vs.enable_hdf5_gzip_compression else {}
            group.require_dataset(key, var.shape, var.dtype, exact=True, **kwargs)
            group[key][...] = var
        for key, val in attributes.items():
            group.attrs[key] = val
