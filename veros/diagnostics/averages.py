from collections import namedtuple
import json
import logging

from .diagnostic import VerosDiagnostic
from . import io_tools
from .. import veros_class_method

Running_sum = namedtuple("Running_sum", ("var", "sum"))

class Averages(VerosDiagnostic):
    """Time average output
    """
    outfile = "{identifier}_averages.nc"
    output_variables = ("psi",)

    @veros_class_method
    def initialize(self, veros):
        """
        register all variables to be averaged
        """
        self.filename = self.outfile.format(**vars(veros))
        self.average_nitts = 0
        self.average_vars = {}
        for var in self.output_variables:
            var_array = getattr(veros, var)
            var_data = veros.variables[var]
            self.average_vars[var] = Running_sum(var_data, np.zeros_like(var_array))
        self.restart_attributes = ("average_nitts",)
        self.restart_arrays = {key: lambda: getattr(self,arr) for key, arr in self.average_vars.items()}

    def diagnose(self, veros):
        self.average_nitts += 1
        for key, var in self.average_vars.items():
            var.sum[...] += getattr(veros, key)

    def output(self, veros):
        """
        write averages to netcdf file and zero array
        """
        with io_tools.threaded_io(veros, self.filename, "w") as f:
            logging.info(" writing averages to file {}".format(self.filename))
            io_tools.initialize_file(veros, f)
            for key, runsum in self.average_vars.items():
                io_tools.initialize_variable(veros, key, runsum.var, f)
                runsum.sum[...] /= self.average_nitts
                io_tools.write_variable(veros, key, runsum.var, 0, f, var_data=runsum.sum)
                runsum.sum[...] = 0.
        self.average_nitts = 0
