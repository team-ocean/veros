import logging
import os
import warnings

from . import io_tools
from .. import veros_class_method, time
from .diagnostic import VerosDiagnostic


class Snapshot(VerosDiagnostic):
    """Writes snapshots of the current solution. Also reads and writes the main restart
    data required for restarting a Veros simulation.
    """
    output_path = "{identifier}.snapshot.nc"
    """File to write to. May contain format strings that are replaced with Veros attributes.

    .. warning::

        If this path is constant between iterations (default), data will be appended to
        the snapshot file. However, upon starting a run, this file will be overwritten.
        This also applies when doing a restart.
        
    """
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    #: Attributes to be written to restart file.
    restart_attributes = ("itt", "tau", "taum1", "taup1")

    def __init__(self, veros):
        self.output_variables = [key for key, val in veros.variables.items() if val.output]
        """Variables to be written to output. Defaults to all Veros variables that
        have the attribute :attr:`output`."""
        self.restart_variables = [key for key,
                                  val in veros.variables.items() if val.write_to_restart]
        """Variables to be written to restart. Defaults to all Veros variables that
        have the attribute :attr:`write_to_restart`."""

    @veros_class_method
    def initialize(self, veros):
        var_meta = {var: veros.variables[var] for var in self.output_variables}
        var_data = {var: getattr(veros, var) for var in self.output_variables}
        self.initialize_output(veros, var_meta, var_data)

    def diagnose(self, veros):
        pass

    @veros_class_method
    def output(self, veros):
        current_time = time.current_time(veros)
        logging.info(" writing snapshot at {0[0]} {0[1]}".format(
            time.format_time(veros, current_time)))

        if not os.path.isfile(self.get_output_file_name(veros)):
            self.initialize(veros)

        var_meta = {var: veros.variables[var]
                    for var in self.output_variables if veros.variables[var].time_dependent}
        var_data = {var: getattr(veros, var) for var in var_meta.keys()}
        self.write_output(veros, var_meta, var_data)

    def read_restart(self, veros):
        restart_vars = {var: veros.variables[var] for var in self.restart_variables}
        restart_data = {var: getattr(veros, var) for var in self.restart_variables}
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
        for attr in self.restart_attributes:
            try:
                setattr(veros, attr, attributes[attr])
            except KeyError:
                warnings.warn("not reading restart data for attribute {}: "
                              "attribute not found in restart file"
                              .format(attr))

    def write_restart(self, veros):
        restart_attributes = {key: getattr(veros, key) for key in self.restart_attributes}
        restart_vars = {var: veros.variables[var] for var in self.restart_variables}
        restart_data = {var: getattr(veros, var) for var in self.restart_variables}
        self.write_h5_restart(veros, restart_attributes, restart_vars, restart_data)
