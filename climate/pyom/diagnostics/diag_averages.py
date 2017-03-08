import warnings
from collections import namedtuple
import os
import json
from netCDF4 import Dataset

from climate.pyom import pyom_method
from climate.pyom.diagnostics.diag_snap import def_grid_cdf
from climate.pyom.diagnostics.io_threading import threaded_netcdf


Diagnostic = namedtuple("Diagnostic", ["name", "long_name", "units", "grid", "var", "sum"])

@pyom_method
def register_average(pyom,name,long_name,units,grid,var):
    """
    register a variables to be averaged
    this routine may be called by user in set_diagnostics
    name : NetCDF variables name (must be unique)
    long_name:  long name
    units : units
    grid : two or three digits, either 'T' for T grid or 'U' for shifted grid
           U is shifted by 1/2 grid point
    var : callable returning the variable to be averaged
    """
    if not callable(var):
        raise TypeError("var must be a callable that returns the variable to be averaged")

    if len(grid) != var().ndim:
        raise ValueError("number of dimensions must match grid")

    if not set(grid) <= {"U","T"}:
        raise ValueError("grid variable may only contain 'U' or 'T'")

    if not pyom.enable_diag_averages:
        warnings.warn("switch on enable_diag_averages to use time averaging")
        return

    print(" time averaging variable {}".format(name))
    print(" long name {} units {} grid {}".format(long_name,units,grid))

    if name.strip() in (d.name.strip() for d in pyom.diagnostics):
        raise RuntimeError("name {} already in use".format(name))

    diag = Diagnostic(name, long_name, units, grid, var, np.zeros_like(var()))
    pyom.diagnostics.append(diag)

@pyom_method
def diag_averages(pyom):
    pyom.average_nitts += 1
    for diag in pyom.diagnostics:
        diag.sum[...] += diag.var()

@pyom_method
def _build_dimensions(pyom, grid):
    if not len(grid) in [2,3]:
        raise ValueError("Grid string must consist of 2 or 3 characters")
    if not set(grid) <= {"U","T"}:
        raise ValueError("Grid string may only consist of 'U' or 'T'")
    dim_x = "xu" if grid[0] == "U" else "xt"
    dim_y = "yu" if grid[1] == "U" else "yt"
    if len(grid) == 3:
        dim_z = "zw" if grid[2] == 'U' else "zt"
        return dim_x, dim_y, dim_z, "Time"
    else:
        return dim_x, dim_y, "Time"

@pyom_method
def _get_mask(pyom, grid):
    if not len(grid) in [2,3]:
        raise ValueError("Grid string must consist of 2 or 3 characters")
    if not set(grid) <= {"U","T"}:
        raise ValueError("Grid string may only consist of 'U' or 'T'")
    masks = {
        "TT": pyom.maskT[:,:,-1],
        "UT": pyom.maskU[:,:,-1],
        "TU": pyom.maskV[:,:,-1],
        "UU": pyom.maskZ[:,:,-1],
        "TTT": pyom.maskT,
        "UTT": pyom.maskU,
        "TUT": pyom.maskV,
        "TTU": pyom.maskW,
        "UUT": pyom.maskZ,
    }
    return masks[grid].astype(np.bool)

@pyom_method
def write_averages(pyom):
    """
    write averages to netcdf file and zero array
    """
    fill_value = -1e33
    filename = "averages_{}.cdf".format(pyom.itt)

    with threaded_netcdf(pyom, Dataset(filename,mode='w')) as f:
        print(" writing averages to file " + filename)
        def_grid_cdf(pyom, f)
        for diag in pyom.diagnostics:
            dims = _build_dimensions(pyom,diag.grid)
            mask = _get_mask(pyom,diag.grid)
            var_data = np.where(mask, diag.sum, np.nan)
            if pyom.backend_name == "bohrium":
                var_data = var_data.copy2numpy()
            var = f.createVariable(diag.name, "f8", dims, fill_value=fill_value)
            var[...] = var_data[2:-2, 2:-2, ..., None] / pyom.average_nitts
            var.long_name = diag.long_name
            var.units = diag.units
            var.missing_value = fill_value
            diag.sum[...] = 0.
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
