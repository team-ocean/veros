import logging
import os
import warnings

from . import io_tools
from .. import veros_class_method, time
from .diagnostic import VerosDiagnostic

RESTART_ATTRIBUTES = {"itt", "tau", "taum1", "taup1"}

class Snapshot(VerosDiagnostic):
    output_path = "{identifier}_snapshot.nc"
    output_frequency = None

    @veros_class_method
    def initialize(self, veros):
        """
        initialize NetCDF snapshot file
        """
        var_meta = {key: val for key, val in veros.variables.items() if val.output}
        var_data = {key: getattr(veros, key) for key, val in veros.variables.items() if val.output}
        self.initialize_output(veros, var_meta, var_data)

    def diagnose(self, veros):
        pass

    @veros_class_method
    def output(self, veros):
        current_time = time.current_time(veros)
        logging.info(" writing snapshot at {0[0]} {0[1]}".format(time.format_time(veros, current_time)))

        if not os.path.isfile(self.get_output_file_name(veros)):
            self.initialize(veros)

        var_meta = {key: val for key, val in veros.variables.items() if val.time_dependent and val.output}
        var_data = {key: getattr(veros, key) for key, val in veros.variables.items() if val.time_dependent and val.output}
        self.write_output(veros, var_meta, var_data)

    def read_restart(self, veros):
        restart_vars = {key: var for key, var in veros.variables.items() if var.write_to_restart}
        restart_data = {key: getattr(veros, key) for key in restart_vars.keys()}
        attributes, variables = self.read_h5_restart(veros)
        for key, arr in restart_data.items():
            try:
                restart_var = variables[key]
            except KeyError:
                warnings.warn("not reading restart data for variable {}: "
                              "no matching data found in restart file"
                              .format(key))
                continue
            if not arr.shape == restart_var.shape:
                warnings.warn("not reading restart data for variable {}: "
                              "restart data dimensions do not match model "
                              "grid".format(key))
                continue
            arr[...] = restart_var
        for attr in RESTART_ATTRIBUTES:
            try:
                setattr(veros, attr, attributes[attr])
            except KeyError:
                warnings.warn("not reading restart data for attribute {}: "
                              "attribute not found in restart file"
                              .format(attr))

    def write_restart(self, veros):
        restart_attributes = {key: getattr(veros, key) for key in RESTART_ATTRIBUTES}
        restart_vars = {key: var for key, var in veros.variables.items() if var.write_to_restart}
        restart_data = {key: getattr(veros, key) for key in restart_vars.keys()}
        self.write_h5_restart(veros, restart_attributes, restart_vars, restart_data)
