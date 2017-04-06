from collections import namedtuple
import json
import logging

from . import io_tools
from .. import veros_method

Running_sum = namedtuple("Running_sum", ("var", "sum"))

@veros_method
def initialize(veros):
    """
    register all variables to be averaged
    """
    veros._average_nitts = 0
    veros._average_vars = {}
    for key, var in veros.variables.items():
        if var.average:
            veros._average_vars[key] = Running_sum(var, np.zeros_like(getattr(veros, key)))

@veros_method
def diagnose(veros):
    veros._average_nitts += 1
    for key, var in veros._average_vars.items():
        var.sum[...] += getattr(veros, key)

@veros_method
def output(veros):
    """
    write averages to netcdf file and zero array
    """
    filename = veros.diagnostics["averages"].outfile.format(**vars(veros))
    with io_tools.threaded_io(veros, filename, "w") as f:
        logging.info(" writing averages to file " + filename)
        io_tools.initialize_file(veros, f)
        for key, runsum in veros._average_vars.items():
            io_tools.initialize_variable(veros, key, runsum.var, f)
            runsum.sum[...] /= veros._average_nitts
            io_tools.write_variable(veros, key, runsum.var, 0, f, var_data=runsum.sum)
            runsum.sum[...] = 0.
    veros._average_nitts = 0


UNFINISHED_AVERAGES_FILENAME = "unfinished_averages.json"

@veros_method
def diag_averages_write_restart(veros):
    """
    write unfinished averages to restart file
    """
    with open(filename,"w") as f:
        print(" writing unfinished averages to {}".format(UNFINISHED_AVERAGES_FILENAME))
        if not os.path.isfile(UNFINISHED_AVERAGES_FILENAME):
            warnings.warn("could not write restart file")
            return
        json_data = dict(nx=veros.nx, ny=veros.ny, nz=veros.nz, nitts=veros.average_nitts)
        json_data["averages"] = dict()
        for diag in veros.diagnostics:
            json_data["averages"][diag.name] = diag.sum.tolist()
        json.dump(json_data,f)

@veros_method
def diag_averages_read_restart(veros):
    """
    read unfinished averages from file
    """
    if not os.path.isfile(UNFINISHED_AVERAGES_FILENAME):
        warnings.warn("file {} not present, reading no unfinished time averages".format(UNFINISHED_AVERAGES_FILENAME))
        return
    print(" reading unfinished averages from {}".format(UNFINISHED_AVERAGES_FILENAME))

    with open(filename,"r") as f:
        json_data = json.load(f)

    if json_data["nx"] != veros.nx or json_data["ny"] != veros.ny or json_data["nz"] != veros.nz:
        warnings.warn("error reading restart file: read dimensions"
                      "{} {} {} do not match dimensions {} {} {}" \
                     .format(json_data["nx"],json_data["ny"],json_data["nz"],
                             veros.nx, veros.ny, veros.nz))
        return

    for diag in veros.diagnostics:
        diag.sum[...] = np.array(json_data["averages"][diag.name])
    veros.average_nitts = json_data["nitts"]
