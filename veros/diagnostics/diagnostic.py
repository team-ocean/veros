from collections import OrderedDict
import os
import copy
import logging

from . import io_tools
from .. import veros_class_method, veros_method, time
from ..variables import Variable, TIMESTEPS

@veros_method
def initialize_restart_file(veros):
    filepath = veros.restart_output_filename.format(**vars(veros))
    with io_tools.threaded_io(veros, filepath, "w") as outfile:
        io_tools.initialize_file(veros, outfile)
        io_tools.add_dimension(veros, TIMESTEPS[0], 3, outfile)

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
        with io_tools.threaded_io(veros, restart_filename, "r") as infile:
            group = infile.groups[self.diagnostic_name]
            variables = io_tools.get_all_variables(veros, group)
            attributes = io_tools.get_all_attributes(veros, group)
        return attributes, variables

    @veros_class_method
    def write_h5_restart(self, veros, attributes, var_meta, var_data, extra_dimensions=None):
        restart_filename = self.get_restart_output_file_name(veros)
        with io_tools.threaded_io(veros, restart_filename, "a") as outfile:
            group = io_tools.create_group(veros, self.diagnostic_name, outfile)
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    io_tools.add_dimension(veros, dim_id, size, group)
            for key, var in var_meta.items():
                var = copy.copy(var)
                var.time_dependent = False
                io_tools.initialize_variable(veros, key, var, outfile, group=group)
                io_tools.write_variable(veros, key, var, var_data[key], outfile,
                                        group=group, strip_time_steps=False, fill=False)
            for key, val in attributes.items():
                io_tools.write_attribute(veros, key, val, group)

    @veros_class_method
    def get_output_file_name(self, veros):
        return self.output_path.format(**vars(veros))

    @veros_class_method
    def initialize_output(self, veros, variables, var_data=None, extra_dimensions=None, filepath=None):
        with io_tools.threaded_io(veros, filepath or self.get_output_file_name(veros), "w") as outfile:
            io_tools.initialize_file(veros, outfile)
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    io_tools.add_dimension(veros, dim_id, size, outfile)
            for key, var in variables.items():
                io_tools.initialize_variable(veros, key, var, outfile)
                if not var.time_dependent:
                    if var_data is None or not key in var_data:
                        raise ValueError("var_data argument must be given for constant variables")
                    io_tools.write_variable(veros, key, var, var_data[key], outfile)

    @veros_class_method
    def write_output(self, veros, variables, variable_data, filepath=None):
        with io_tools.threaded_io(veros, filepath or self.get_output_file_name(veros), "a") as outfile:
            time_step = io_tools.get_current_timestep(veros, outfile)
            io_tools.advance_time(veros, time_step, time.current_time(veros, "days"), outfile)
            for key, var in variables.items():
                io_tools.write_variable(veros, key, var, variable_data[key],
                                        outfile, time_step=time_step)


    def _not_implemented(self, veros):
        raise NotImplementedError("must be implemented by subclass")

    initialize = _not_implemented
    diagnose = _not_implemented
    output = _not_implemented
    write_restart = _not_implemented
    read_restart = _not_implemented
