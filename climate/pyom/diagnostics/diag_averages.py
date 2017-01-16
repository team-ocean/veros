import warnings
from collections import namedtuple
from netCDF4 import Dataset
import json

from climate.pyom.diagnostics.diag_snap import def_grid_cdf

# module diag_averages_module
# """
# ! Module for time averages
# """
#   implicit none
#   integer :: nitts = 0,number_diags = 0
#   integer, parameter :: max_number_diags = 500
#   type type_var2D
#     real*8,pointer  :: a(:,:)
#   end type type_var2D
#   type type_var3D
#     real*8,pointer  :: a(:,:,:)
#   end type type_var3D
#   character (len = 80) :: diag_name(max_number_diags), diag_longname(max_number_diags)
#   character (len = 80) :: diag_units(max_number_diags), diag_grid(max_number_diags)
#   type(type_var2d)   :: diag_var2D(max_number_diags), diag_sum_var2D(max_number_diags)
#   type(type_var3d)   :: diag_var3D(max_number_diags), diag_sum_var3D(max_number_diags)
#   logical            :: diag_is3D(max_number_diags)
# end module diag_averages_module


Diagnostic = namedtuple("Diagnostic", ["name", "long_name", "units", "grid", "var", "sum"])


def register_average(name,long_name,units,grid,var,pyom):
    """
    register a variables to be averaged
    this routine may be called by user in set_diagnostics
    name : NetCDF variables name (must be unique)
    long_name:  long name
    units : units
    grid : three digits, either 'T' for T grid or 'U' for shifted grid
           applies for the 3 dimensions, U is shifted by 1/2 grid point
    var : variable to be averaged
    """
    if len(grid) != var.ndim:
        if len(grid)+1 == var.ndim:
            var_slice = tuple([None]*len(grid) + [-1])
        else:
            raise ValueError("number of dimensions must match grid")
    else:
        var_slice = tuple([None]*len(grid))
    if not all([g == "U" or g == "T" for g in grid]):
        raise ValueError("grid variable may only contain 'U' or 'T'")

    if not pyom.enable_diag_averages:
        warnings.warn("switch on enable_diag_averages to use time averaging")
        return

    print(" time averaging variable {}".format(name))
    print(" long name {} units {} grid {}".format(long_name,units,grid))
    # check if name is in use
    for diag in pyom.diags: # n = 1,number_diags
        if name.strip() == diag["name"].strip():
            raise RuntimeError(" name already in use")

    diag = Diagnostic(name, long_name, units, grid, var[var_slice].view, np.zeros_like(var[var_slice]))
    pyom.diagnostics.append(diag)

def diag_averages(pyom):
    for diag in pyom.diagnostics:
        diag.sum[...] += diag.var[...]

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
            var_data = np.copy(diag.sum)
            diag.sum[...] = 0.
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
                         nitts=pyom.nitts, number_diags=pyom.number_diags)
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
        warnings.warn(" file {} not present\n"
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

    for diag in diagnostics:
        diag.sum[...] = np.array(json_data["averages"][diag.name])
