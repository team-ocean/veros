import warnings
from collections import namedtuple
from netCDF4 import Dataset
import json
import numpy as np

from climate.pyom.diagnostics.diag_snap import def_grid_cdf


Diagnostic = namedtuple("Diagnostic", ["name", "long_name", "units", "grid", "var", "sum"])

def register_average(name,long_name,units,grid,var,pyom):
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

    if not all([g == "U" or g == "T" for g in grid]):
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

def diag_averages(pyom):
    for diag in pyom.diagnostics:
        diag.sum[...] += diag.var()[...]

def write_averages(pyom):
    """
    write averages to netcdf file and zero array
    """
    # real*8, parameter :: spval = -1.0d33
    # real*8 :: bloc(nx,ny),fxa

    fill_value = -1e33
    filename = "averages_{}.cdf".format(pyom.itt)
    with Dataset(filename,mode='a') as f:
        print(" writing averages to file " + f)
        def_grid_cdf(f)
        f.set_fill_off()
        for diag in pyom.diagnostics: # n=1,number_diags
            dim_x = "xu" if diag.grid[0] == 'U' else "xt"
            dim_y = "yu" if diag.grid[1] == 'U' else "yt"
            if len(diag.grid) == 3:
                dim_z = "zu" if diag.grid[2] == 'U' else "zt"
                dims = (dim_x, dim_y, dim_z, "Time")
            else:
                dims = (dim_x, dim_y, "Time")
            var = f.createVariable(diag.name,"f8",dims,fill_value=fill_value)
            var_data = diag.sum
            if diag.grid[:2] == "TU":
                mask = pyom.maskV
            elif diag.grid[:2] == "UT":
                mask = pyom.maskU
            else:
                if diag.grid[2] == "T":
                    mask = pyom.maskT
                elif diag.grid[2] == "U":
                    mask = pyom.maskU
            var_data[mask == 0] = np.nan
            var[:] = var_data
            var.long_name = diag.long_name
            var.units = diag.units
            var.missing_value = fill_value
            diag.sum[...] = 0.

def diag_averages_write_restart(pyom):
    """
    write unfinished averages to restart file
    """
    filename = "unfinished_averages.json"
    with open(filename,"w") as f:
        print(" writing unfinished averages to {}".format(filename))
        if not os.file_exists(filename):
            warnings.warn("could not write restart file")
            return
        json_data = dict(nx=pyom.nx, ny=pyom.ny, nz=pyom.nz,
                         nitts=pyom.nitts)
        json_data["averages"] = dict()
        for diag in pyom.diagnostics:
            json_data["averages"][diag.name] = diag.sum.tolist()
        json.dump(json_data,f)

def diag_averages_read_restart(pyom):
    """
    read unfinished averages from file
    """
    filename = "unfinished_averages.json"
    if not os.file_exists(filename):
        warnings.warn("file {} not present\n"
                      "reading no unfinished time averages".format(filename))
        return
    print(" reading unfinished averages from {}".format(filename))

    with open(filename,"r") as f:
        json_data = json.load(f)

    if json_data["nx"] != pyom.nx or json_data["ny"] != pyom.ny or json_data["nz"] != pyom.nz:
        warnings.warn("error reading restart file: read dimensions"
                      "{} {} {} do not match dimensions {} {} {}"\
                     .format(json_data["nx"],json_data["ny"],json_data["nz"],
                             pyom.nx, pyom.ny, pyom.nz))
        return

    for diag in pyom.diagnostics:
        diag.sum[...] = np.array(json_data["averages"][diag.name])
