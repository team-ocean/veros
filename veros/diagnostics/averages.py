import os
import copy

from veros.diagnostics.base import VerosDiagnostic
from veros.variables import TIMESTEPS, Variable


class Averages(VerosDiagnostic):
    """Time average output diagnostic.

    All registered variables are summed up when :meth:`diagnose` is called,
    and averaged and output upon calling :meth:`output`.
    """

    name = "averages"  #:
    output_path = "{identifier}.averages.nc"  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_variables = None  #: Iterable containing all variables to be averaged. Changes have no effect after ``initialize`` has been called.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.

    def __init__(self, state):
        self.var_meta = {
            "average_nitts": Variable("average_nitts", None, write_to_restart=True),
        }
        self.output_variables = []

    def initialize(self, state):
        """Register all variables to be averaged"""

        for var in self.output_variables:
            var_meta = copy.copy(state.var_meta[var])
            var_meta.time_dependent = True
            var_meta.write_to_restart = True

            if self._has_timestep_dim(state, var):
                var_meta.dims = var_meta.dims[:-1]

            self.var_meta[var] = var_meta

        self.initialize_variables(state)
        self.initialize_output(state)

    @staticmethod
    def _has_timestep_dim(state, var):
        if state.var_meta[var].dims is None:
            return False

        return state.var_meta[var].dims[-1] == TIMESTEPS[0]

    def diagnose(self, state):
        vs = state.variables
        avg_vs = self.variables

        avg_vs.average_nitts = avg_vs.average_nitts + 1

        for key in self.output_variables:
            var_data = getattr(avg_vs, key)
            if self._has_timestep_dim(state, key):
                setattr(avg_vs, key, var_data + getattr(vs, key)[..., vs.tau])
            else:
                setattr(avg_vs, key, var_data + getattr(vs, key))

    def output(self, state):
        """Write averages to netcdf file and zero array"""
        avg_vs = self.variables

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        if avg_vs.average_nitts > 0:
            for key in self.output_variables:
                val = getattr(avg_vs, key)
                setattr(avg_vs, key, val / avg_vs.average_nitts)

        self.write_output(state)

        for key in self.output_variables:
            val = getattr(avg_vs, key)
            setattr(avg_vs, key, 0 * val)

        avg_vs.average_nitts = 0
