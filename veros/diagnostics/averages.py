from collections import namedtuple
import json
import logging
import os

from .diagnostic import VerosDiagnostic
from . import io_tools
from .. import veros_class_method
from ..variables import Variable

Running_sum = namedtuple("Running_sum", ("var", "sum"))

class Averages(VerosDiagnostic):
    """Time average output diagnostic.

    All registered variables are summed up when :meth:`diagnose` is called,
    and averaged and output upon calling :meth:`output`.
    """
    output_path = "{identifier}_averages.nc" #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_variables = None #: Iterable containing all variables to be averaged. Changes have no effect after ``initialize`` has been called.
    output_frequency = None #: Frequency (in seconds) in which output is written.
    sampling_frequency = None #: Frequency (in seconds) in which variables are accumulated.

    @veros_class_method
    def initialize(self, veros):
        """Register all variables to be averaged
        """
        self.average_nitts = 0
        self.average_vars = {}
        if not self.output_variables:
            return
        for var in self.output_variables:
            var_array = getattr(veros, var)
            var_data = veros.variables[var]
            var_data.time_dependent = True
            self.average_vars[var] = Running_sum(var_data, np.zeros_like(var_array))
        self.initialize_output(veros, {key: runsum.var for key, runsum in self.average_vars.items()})

    def diagnose(self, veros):
        self.average_nitts += 1
        for key, var in self.average_vars.items():
            var.sum[...] += getattr(veros, key)

    def output(self, veros):
        """Write averages to netcdf file and zero array
        """
        variable_metadata = {key: runsum.var for key, runsum in self.average_vars.items()}
        if not os.path.isfile(self.get_output_file_name(veros)):
            self.initialize_output(veros, variable_metadata)
        variable_mean = {key: runsum.sum / self.average_nitts for key, runsum in self.average_vars.items()}
        self.write_output(veros, variable_metadata, variable_mean)
        for runsum in self.average_vars.values():
            runsum.sum[...] = 0.
        self.average_nitts = 0

    def read_restart(self, veros):
        attributes, variables = self.read_h5_restart(veros)
        if attributes:
            self.average_nitts = attributes["average_nitts"]
        if variables:
            self.average_vars = {key: Running_sum(veros.variables[key], var) for key, var in variables.items()}

    def write_restart(self, veros):
        attributes = {"average_nitts": self.average_nitts}
        variables = {key: runsum.sum for key, runsum in self.average_vars.items()}
        variable_metadata = {key: runsum.var for key, runsum in self.average_vars.items()}
        self.write_h5_restart(veros, attributes, variable_metadata, variables)
