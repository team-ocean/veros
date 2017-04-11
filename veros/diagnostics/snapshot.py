import logging
import os

from . import io_tools
from .. import veros_class_method, time
from .diagnostic import VerosDiagnostic

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
        if time.current_time(veros, "days") < 1:
            logging.info(" writing snapshot at {}s".format(time.current_time(veros, "seconds")))
        else:
            logging.info(" writing snapshot at {}d".format(time.current_time(veros, "days")))

        if not os.path.isfile(self.get_output_file_name(veros)):
            self.initialize(veros)

        var_meta = {key: val for key, val in veros.variables.items() if val.time_dependent and val.output}
        var_data = {key: getattr(veros, key) for key, val in veros.variables.items() if val.time_dependent and val.output}
        self.write_output(veros, var_meta, var_data)

    def read_restart(self, veros):
        pass

    def write_restart(self, veros):
        pass
