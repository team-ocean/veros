import abc

import os

from veros.io_tools import netcdf as nctools
from veros.signals import do_not_disturb
from veros.state import VerosVariables
from veros import distributed, runtime_settings, time


class VerosDiagnostic(metaclass=abc.ABCMeta):
    """Diagnostic interface with shared initialization and output helpers."""

    name = None  #: Name that identifies the current diagnostic
    sampling_frequency = 0.0
    output_frequency = 0.0

    output_path = None
    output_variables = None

    var_meta = None  #: Metadata of internal variables
    extra_dimensions = None  #: Dict of extra dimensions used in var_meta
    supports_compilation = False  #: Numerical methods support JAX tracing.
    observes_model = False  #: Diagnostic methods leave model variables and settings unchanged.

    def __init__(self, state):
        pass

    @abc.abstractmethod
    def initialize(self, state):
        """Called at the end of setup. Use this to process user settings and handle setup."""
        pass

    @abc.abstractmethod
    def diagnose(self, state):
        """Called with frequency ``sampling_frequency``."""
        pass

    @abc.abstractmethod
    def output(self, state):
        """Called with frequency ``output_frequency``."""
        pass

    def finalize(self, state):
        """Return numerical output and reset its accumulation window."""
        raise NotImplementedError(f"Diagnostic {self.name} does not implement numerical finalization")

    def write_finalized(self, state, values, time_value=None):
        """Write a finalized payload without changing accumulators."""
        if runtime_settings.diskless_mode:
            return
        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)
        self.write_output(state, values, time_value=time_value, time_indexed=True)

    def initialize_variables(self, state):
        if self.var_meta is None:
            self.variables = None
            return

        dimensions = dict(state.dimensions)

        if self.extra_dimensions is not None:
            dimensions.update(self.extra_dimensions)

        self.variables = VerosVariables(self.var_meta, dimensions)

        # we leave diagnostic variables unlocked
        self.variables.__locked__ = False

    def get_output_file_name(self, state):
        statedict = dict(state.variables.items())
        statedict.update(state.settings.items())
        return self.output_path.format(**statedict)

    @do_not_disturb
    def initialize_output(self, state):
        inactive = not self.output_frequency and not self.sampling_frequency
        no_output = not self.output_path or not self.output_variables

        if runtime_settings.diskless_mode or inactive or no_output:
            return

        output_path = self.get_output_file_name(state)
        if os.path.isfile(output_path) and not runtime_settings.force_overwrite:
            raise IOError(
                f'output file {output_path} for diagnostic "{self.name}" exists '
                "(change output path or enable force_overwrite runtime setting)"
            )

        # possible race condition ahead!
        distributed.barrier()

        with nctools.threaded_io(output_path, "w") as outfile:
            nctools.initialize_file(state, outfile, extra_dimensions=self.extra_dimensions)

            for key in self.output_variables:
                var = self.var_meta[key]
                if key not in outfile.variables:
                    nctools.initialize_variable(state, key, var, outfile)

                if not var.time_dependent:
                    var_data = self.variables.get(key)
                    nctools.write_variable(state, key, var, var_data, outfile)

    @do_not_disturb
    def write_output(self, state, values=None, *, time_value=None, time_indexed=False):
        vs = state.variables

        if runtime_settings.diskless_mode:
            return

        with nctools.threaded_io(self.get_output_file_name(state), "r+") as outfile:
            current_days = time.convert_time(vs.time if time_value is None else time_value, "seconds", "days")
            nctools.advance_time(current_days, outfile)

            for key in self.output_variables:
                var = self.var_meta[key]
                var_data = self.variables.get(key) if values is None else values[key]
                nctools.write_variable(state, key, var, var_data, outfile, time_indexed=time_indexed)
