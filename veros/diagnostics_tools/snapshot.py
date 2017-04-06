import logging

from . import io_tools
from .. import veros_method

@veros_method
def initialize(veros):
    """
    initialize NetCDF snapshot file
    """
    filename = veros.diagnostics["snapshot"].outfile.format(**vars(veros))
    logging.info("Preparing file {}".format(filename))
    with io_tools.threaded_io(veros, filename, "w") as snap_dataset:
        io_tools.initialize_file(veros, snap_dataset)
        for key, var in veros.variables.items():
            if var.output:
                io_tools.initialize_variable(veros,key,var,snap_dataset)

def diagnose(veros):
    pass

@veros_method
def output(veros):
    filename = veros.diagnostics["snapshot"].outfile.format(**vars(veros))
    time_in_days = veros.itt * veros.dt_tracer / 86400.
    if time_in_days < 1.0:
        logging.info(" writing snapshot at {}s".format(time_in_days * 86400.))
    else:
        logging.info(" writing snapshot at {}d".format(time_in_days))

    with io_tools.threaded_io(veros, filename, "a") as snap_dataset:
        snapshot_number = snap_dataset["Time"].size
        snap_dataset["Time"][snapshot_number] = time_in_days
        for key, var in veros.variables.items():
            if var.output and var.time_dependent:
                io_tools.write_variable(veros, key, var, snapshot_number, snap_dataset)
