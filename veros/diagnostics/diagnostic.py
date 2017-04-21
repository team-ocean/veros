from collections import OrderedDict
import os
import copy
import logging

from .io_tools import netcdf as nctools
from .io_tools import hdf5 as h5tools
from .. import veros_class_method, veros_method, time
from ..variables import Variable


class VerosDiagnostic(object):
    sampling_frequency = 0.
    output_frequency = 0.
    output_path = None

    @property
    def diagnostic_name(self):
        return self.__class__.__name__

    @veros_class_method
    def get_restart_input_file_name(self, veros):
        return veros.restart_input_filename.format(**vars(veros))

    @veros_class_method
    def get_restart_output_file_name(self, veros):
        return veros.restart_output_filename.format(**vars(veros))

    @veros_class_method
    def read_h5_restart(self, veros):
        restart_filename = self.get_restart_input_file_name(veros)
        if not os.path.isfile(restart_filename):
            raise IOError("restart file {} not found".format(restart_filename))

        logging.info("reading restart data for diagnostic {} from {}".format(self.diagnostic_name, restart_filename))
        with h5tools.threaded_io(veros, restart_filename, "r") as infile:
            variables = {key: var[...] for key, var in infile[self.diagnostic_name].items()}
            attributes = {key: var for key, var in infile[self.diagnostic_name].attrs.items()}
        return attributes, variables

    @veros_class_method
    def write_h5_restart(self, veros, attributes, var_meta, var_data):
        restart_filename = self.get_restart_output_file_name(veros)
        with h5tools.threaded_io(veros, restart_filename, "a") as outfile:
            outfile.require_group(self.diagnostic_name)
            for key, var in var_data.items():
                var_name = "{}/{}".format(self.diagnostic_name, key)
                outfile.require_dataset(var_name, var.shape, var.dtype, exact=True)
                outfile[var_name][...] = var_data[key]
            for key, val in attributes.items():
                outfile[self.diagnostic_name].attrs[key] = val

    @veros_class_method
    def get_output_file_name(self, veros):
        return self.output_path.format(**vars(veros))

    @veros_class_method
    def initialize_output(self, veros, variables, var_data=None, extra_dimensions=None, filepath=None):
        with nctools.threaded_io(veros, filepath or self.get_output_file_name(veros), "w") as outfile:
            nctools.initialize_file(veros, outfile)
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    nctools.add_dimension(veros, dim_id, size, outfile)
            for key, var in variables.items():
                if not key in outfile.variables:
                    nctools.initialize_variable(veros, key, var, outfile)
                if not var.time_dependent:
                    if var_data is None or not key in var_data:
                        raise ValueError("var_data argument must be given for constant variables")
                    nctools.write_variable(veros, key, var, var_data[key], outfile)

    @veros_class_method
    def write_output(self, veros, variables, variable_data, filepath=None):
        with nctools.threaded_io(veros, filepath or self.get_output_file_name(veros), "r+") as outfile:
            time_step = nctools.get_current_timestep(veros, outfile)
            nctools.advance_time(veros, time_step, time.current_time(veros, "days"), outfile)
            for key, var in variables.items():
                nctools.write_variable(veros, key, var, variable_data[key],
                                        outfile, time_step=time_step)


    def _not_implemented(self, veros):
        raise NotImplementedError("must be implemented by subclass")

    initialize = _not_implemented
    diagnose = _not_implemented
    output = _not_implemented
    write_restart = _not_implemented
    read_restart = _not_implemented
