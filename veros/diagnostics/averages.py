from collections import namedtuple
import json
import logging
import os
import copy

from .diagnostic import VerosDiagnostic
from . import io_tools
from .. import veros_class_method
from ..variables import Variable, TIMESTEPS

Running_sum = namedtuple("Running_sum", ("var", "sum"))


class Averages(VerosDiagnostic):
    """Time average output diagnostic.

    All registered variables are summed up when :meth:`diagnose` is called,
    and averaged and output upon calling :meth:`output`.
    """
    name = "averages" #:
    output_path = "{identifier}.averages.nc"  #: File to write to. May contain format strings that are replaced with Veros attributes.
    #: Iterable containing all variables to be averaged. Changes have no effect after ``initialize`` has been called.
    output_variables = None #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.

    @veros_class_method
    def initialize(self, vs):
        """Register all variables to be averaged
        """
        self.average_nitts = 0
        self.average_vars = {}

        if not self.output_variables:
            return
        for var in self.output_variables:
            var_data = copy.copy(vs.variables[var])
            var_data.time_dependent = True
            if self._cut_timestep(vs, var):
                var_data.dims = var_data.dims[:-1]
                var_sum = np.zeros_like(getattr(vs, var)[..., vs.tau])
            else:
                var_sum = np.zeros_like(getattr(vs, var))
            self.average_vars[var] = Running_sum(var_data, var_sum)
        self.initialize_output(vs, {key: runsum.var for key,
                                       runsum in self.average_vars.items()})

    @staticmethod
    def _cut_timestep(vs, var):
        return vs.variables[var].dims[-1] == TIMESTEPS[0]

    def diagnose(self, vs):
        self.average_nitts += 1
        for key, var in self.average_vars.items():
            if self._cut_timestep(vs, key):
                var.sum[...] += getattr(vs, key)[..., vs.tau]
            else:
                var.sum[...] += getattr(vs, key)

    def output(self, vs):
        """Write averages to netcdf file and zero array
        """
        variable_metadata = {key: runsum.var for key, runsum in self.average_vars.items()}
        if not os.path.isfile(self.get_output_file_name(vs)):
            self.initialize_output(vs, variable_metadata)
        variable_mean = {key: runsum.sum / self.average_nitts for key,
                         runsum in self.average_vars.items()}
        self.write_output(vs, variable_metadata, variable_mean)
        for runsum in self.average_vars.values():
            runsum.sum[...] = 0.
        self.average_nitts = 0

    def read_restart(self, vs):
        attributes, variables = self.read_h5_restart(vs)
        if attributes:
            self.average_nitts = attributes["average_nitts"]
        if variables:
            self.average_vars = {key: Running_sum(copy.copy(vs.variables[key]), var)
                                for key, var in variables.items()}
        for key, runsum in self.average_vars.items():
            runsum.var.time_dependent = True
            if self._cut_timestep(vs, key):
                runsum.var.dims = runsum.var.dims[:-1]

    def write_restart(self, vs, outfile):
        attributes = {"average_nitts": self.average_nitts}
        variables = {key: runsum.sum for key, runsum in self.average_vars.items()}
        variable_metadata = {key: runsum.var for key, runsum in self.average_vars.items()}
        self.write_h5_restart(vs, attributes, variable_metadata, variables, outfile)
