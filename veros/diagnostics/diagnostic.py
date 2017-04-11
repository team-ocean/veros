from collections import OrderedDict
import os
import logging
import h5py

from . import io_tools
from .. import veros_class_method, time

class VerosDiagnostic(object):
    sampling_frequency = 0.
    output_frequency = 0.
    output_path = None

    @property
    def is_active(self):
        return self.sampling_frequency or self.output_frequency

    @property
    def diagnostic_name(self):
        return self.__class__.__name__

    @veros_class_method
    def get_restart_file_name(self, veros):
        return veros.restart_filename.format(**vars(veros))

    @veros_class_method
    def read_h5_restart(self, veros):
        restart_filename = veros.restart_filename.format(**vars(veros))
        if not os.path.isfile(restart_filename):
            logging.debug("no restart file {} present, not reading restart data"
                          .format(restart_filename))
            return None, None
        logging.info("reading restart data from {}".format(restart_filename))
        group_name = "diagnostics/{}".format(self.diagnostic_name)
        with h5py.File(restart_filename, "r") as restart_file:
            variables = {key: np.array(val) for key, val in restart_file[group_name].items()}
            attributes = {key: val for key, val in restart_file[group_name].attrs.items()}
        return attributes, variables

    def write_h5_restart(self, veros, attributes, variables):
        group_name = "diagnostics/{}".format(self.diagnostic_name)
        with h5py.File(self.get_restart_file_name(veros), "a") as restart_file:
            for key, var_data in variables.items():
                restart_file.create_dataset("{}/{}".format(group_name, key), data=var_data)
            for key, value in attributes.items():
                restart_file[group_name].attrs[key] = value

    @veros_class_method
    def get_output_file_name(self, veros):
        return self.output_path.format(**vars(veros))

    @veros_class_method
    def initialize_output(self, veros, variables, var_data=None, extra_dimensions=None):
        with io_tools.threaded_io(veros, self.get_output_file_name(veros), "w") as outfile:
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
    def write_output(self, veros, variables, variable_data):
        with io_tools.threaded_io(veros, self.get_output_file_name(veros), "a") as outfile:
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
