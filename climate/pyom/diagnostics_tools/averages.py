from collections import namedtuple
import json
import logging

from . import io_tools
from .. import pyom_method

Running_sum = namedtuple("Running_sum", ("var", "sum"))

@pyom_method
def initialize(pyom):
    """
    register all variables to be averaged
    """
    pyom._average_nitts = 0
    pyom._average_vars = {}
    for key, var in pyom.variables.items():
        if var.average:
            pyom._average_vars[key] = Running_sum(var, np.zeros_like(getattr(pyom, key)))

@pyom_method
def diagnose(pyom):
    pyom._average_nitts += 1
    for key, var in pyom._average_vars.items():
        var.sum[...] += getattr(pyom, key)

@pyom_method
def output(pyom):
    """
    write averages to netcdf file and zero array
    """
    filename = pyom.diagnostics["averages"].outfile.format(**vars(pyom))
    with io_tools.threaded_io(pyom, filename, "w") as f:
        logging.info(" writing averages to file " + filename)
        io_tools.initialize_file(pyom, f)
        for key, runsum in pyom._average_vars.items():
            io_tools.initialize_variable(pyom, key, runsum.var, f)
            runsum.sum[...] /= pyom._average_nitts
            io_tools.write_variable(pyom, key, runsum.var, 0, f, var_data=runsum.sum)
            runsum.sum[...] = 0.
    pyom._average_nitts = 0


UNFINISHED_AVERAGES_FILENAME = "unfinished_averages.json"

@pyom_method
def diag_averages_write_restart(pyom):
    """
    write unfinished averages to restart file
    """
    with open(filename,"w") as f:
        print(" writing unfinished averages to {}".format(UNFINISHED_AVERAGES_FILENAME))
        if not os.path.isfile(UNFINISHED_AVERAGES_FILENAME):
            warnings.warn("could not write restart file")
            return
        json_data = dict(nx=pyom.nx, ny=pyom.ny, nz=pyom.nz, nitts=pyom.average_nitts)
        json_data["averages"] = dict()
        for diag in pyom.diagnostics:
            json_data["averages"][diag.name] = diag.sum.tolist()
        json.dump(json_data,f)

@pyom_method
def diag_averages_read_restart(pyom):
    """
    read unfinished averages from file
    """
    if not os.path.isfile(UNFINISHED_AVERAGES_FILENAME):
        warnings.warn("file {} not present, reading no unfinished time averages".format(UNFINISHED_AVERAGES_FILENAME))
        return
    print(" reading unfinished averages from {}".format(UNFINISHED_AVERAGES_FILENAME))

    with open(filename,"r") as f:
        json_data = json.load(f)

    if json_data["nx"] != pyom.nx or json_data["ny"] != pyom.ny or json_data["nz"] != pyom.nz:
        warnings.warn("error reading restart file: read dimensions"
                      "{} {} {} do not match dimensions {} {} {}" \
                     .format(json_data["nx"],json_data["ny"],json_data["nz"],
                             pyom.nx, pyom.ny, pyom.nz))
        return

    for diag in pyom.diagnostics:
        diag.sum[...] = np.array(json_data["averages"][diag.name])
    pyom.average_nitts = json_data["nitts"]
