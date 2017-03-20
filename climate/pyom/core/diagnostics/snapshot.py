import logging

from ...io import netcdf_tools, netcdf_threading
from ... import pyom_method

@pyom_method
def initialize(pyom):
    """
    initialize NetCDF snapshot file
    """
    filename = pyom.diagnostics["snapshot"].outfile.format(**vars(pyom))
    logging.info("Preparing file {}".format(filename))
    with netcdf_threading.threaded_netcdf(pyom, filename, "w") as snap_dataset:
        netcdf_tools.initialize_netcdf_file(pyom, snap_dataset)
        for key, var in pyom.variables.items():
            if var.output:
                netcdf_tools.initialize_variable(pyom,key,var,snap_dataset)

def diagnose(pyom):
    pass

@pyom_method
def output(pyom):
    filename = pyom.diagnostics["snapshot"].outfile.format(**vars(pyom))
    time_in_days = pyom.itt * pyom.dt_tracer / 86400.
    if time_in_days < 1.0:
        logging.info(" writing snapshot at {}s".format(time_in_days * 86400.))
    else:
        logging.info(" writing snapshot at {}d".format(time_in_days))

    with netcdf_threading.threaded_netcdf(pyom, filename, "a") as snap_dataset:
        snapshot_number = snap_dataset["Time"].size
        snap_dataset["Time"][snapshot_number] = time_in_days
        for key, var in pyom.variables.items():
            if var.output and var.time_dependent:
                netcdf_tools.write_variable(pyom, key, var, snapshot_number, snap_dataset)
