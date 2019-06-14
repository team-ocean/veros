import os

from loguru import logger

from .. import veros_method, time
from .diagnostic import VerosDiagnostic


class Snapshot(VerosDiagnostic):
    """Writes snapshots of the current solution. Also reads and writes the main restart
    data required for restarting a Veros simulation.
    """
    output_path = '{identifier}.snapshot.nc'
    """File to write to. May contain format strings that are replaced with Veros attributes."""
    name = 'snapshot' #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    #: Attributes to be written to restart file.
    restart_attributes = ('itt', 'time', 'tau', 'taum1', 'taup1')

    def __init__(self, vs):
        self.output_variables = [key for key, val in vs.variables.items() if val.output]
        """Variables to be written to output. Defaults to all Veros variables that
        have the attribute :attr:`output`."""
        self.restart_variables = [key for key,
                                  val in vs.variables.items() if val.write_to_restart]
        """Variables to be written to restart. Defaults to all Veros variables that
        have the attribute :attr:`write_to_restart`."""

    @veros_method
    def initialize(self, vs):
        var_meta = {var: vs.variables[var] for var in self.output_variables}
        var_data = {var: getattr(vs, var) for var in self.output_variables}
        self.initialize_output(vs, var_meta, var_data)

    def diagnose(self, vs):
        pass

    @veros_method
    def output(self, vs):
        logger.info(' Writing snapshot at {0[0]:.2f} {0[1]}', time.format_time(vs.time))

        if not os.path.isfile(self.get_output_file_name(vs)):
            self.initialize(vs)

        var_meta = {var: vs.variables[var]
                    for var in self.output_variables if vs.variables[var].time_dependent}
        var_data = {var: getattr(vs, var) for var in var_meta.keys()}
        self.write_output(vs, var_meta, var_data)

    def read_restart(self, vs, infile):
        restart_vars = {var: vs.variables[var] for var in self.restart_variables}
        restart_data = {var: getattr(vs, var) for var in self.restart_variables}
        attributes, variables = self.read_h5_restart(vs, restart_vars, infile)
        for key, arr in restart_data.items():
            try:
                restart_var = variables[key]
            except KeyError:
                logger.warning('Not reading restart data for variable {}: '
                               'no matching data found in restart file'
                               .format(key))
                continue
            if not arr.shape == restart_var.shape:
                logger.warning('Not reading restart data for variable {}: '
                               'restart data dimensions do not match model '
                               'grid'.format(key))
                continue
            arr[...] = restart_var
        for attr in self.restart_attributes:
            try:
                setattr(vs, attr, attributes[attr])
            except KeyError:
                logger.warning('Not reading restart data for attribute {}: '
                               'attribute not found in restart file'
                               .format(attr))

    def write_restart(self, vs, outfile):
        restart_attributes = {key: getattr(vs, key) for key in self.restart_attributes}
        restart_vars = {var: vs.variables[var] for var in self.restart_variables}
        restart_data = {var: getattr(vs, var) for var in self.restart_variables}
        self.write_h5_restart(vs, restart_attributes, restart_vars, restart_data, outfile)
