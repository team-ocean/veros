import warnings
import os
import json
from collections import namedtuple

from climate.pyom import pyom_method, variables
from climate.pyom.diagnostics.diag_snap import def_grid_cdf, init_var, write_var
from climate.pyom.diagnostics.io_threading import threaded_netcdf

Running_sum = namedtuple("Running_sum", ("var", "sum"))

@pyom_method
def register_averages(pyom):
    """
    register all variables to be averaged
    """
    pyom.average_nitts = 0
    pyom.average_vars = {}
    for key, var in pyom.variables.items():
        if var.average:
            pyom.average_vars[key] = Running_sum(var, np.zeros_like(getattr(pyom, key)))

@pyom_method
def diag_averages(pyom):
    pyom.average_nitts += 1
    for key, var in pyom.average_vars.items():
        var.sum[...] += getattr(pyom, key)

@pyom_method
def write_averages(pyom):
    """
    write averages to netcdf file and zero array
    """
    filename = pyom.average_filename.format(pyom.itt)
    with threaded_netcdf(pyom, filename, "w") as f:
        print(" writing averages to file " + filename)
        def_grid_cdf(pyom, f)
        for key, runsum in pyom.average_vars.items():
            init_var(pyom, key, runsum.var, f)
            runsum.sum[...] /= pyom.average_nitts
            write_var(pyom, key, runsum.var, 0, f, var_data=runsum.sum)
            runsum.sum[...] = 0.
    pyom.average_nitts = 0


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
