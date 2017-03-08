import os
from collections import namedtuple
from netCDF4 import Dataset

from climate.pyom.diagnostics.io_threading import threaded_netcdf
from climate.pyom import pyom_method

@pyom_method
def def_grid_cdf(pyom,ncfile):
    """
    Define standard grid in netcdf file
    """
    if not isinstance(ncfile,Dataset):
        raise TypeError("Argument needs to be a netCDF4 Dataset")

    # dimensions
    lon_tdim = ncfile.createDimension("xt", pyom.nx)
    lon_udim = ncfile.createDimension("xu", pyom.nx)
    lat_tdim = ncfile.createDimension("yt", pyom.ny)
    lat_udim = ncfile.createDimension("yu", pyom.ny)
    iTimedim = ncfile.createDimension("Time", None)
    isle_dim = ncfile.createDimension("isle", pyom.nisle)
    # grid variables
    lon_tid = ncfile.createVariable("xt","f8",("xt",))
    lon_uid = ncfile.createVariable("xu","f8",("xu",))
    lat_tid = ncfile.createVariable("yt","f8",("yt",))
    lat_uid = ncfile.createVariable("yu","f8",("yu",))
    itimeid = ncfile.createVariable("Time","f8",("Time",))
    # attributes of the grid
    if pyom.coord_degree:
        lon_tid.long_name = "Longitude on T grid"
        lon_tid.units = "degrees E"
        lon_uid.long_name = "Longitude on U grid"
        lon_uid.units = "degrees E"
        lat_tid.long_name = "Latitude on T grid"
        lat_tid.units = "degrees N"
        lat_uid.long_name = "Latitude on U grid"
        lat_tid.units = "degrees N"
    else:
        lon_tid.long_name = "zonal axis T grid"
        lon_tid.units = "km"
        lon_uid.long_name = "zonal axis U grid"
        lon_uid.units = "km"
        lat_tid.long_name = "meridional axis T grid"
        lat_tid.units = "km"
        lat_uid.long_name = "meridional axis U grid"
        lat_tid.units = "km"
    itimeid.long_name = "Time"
    itimeid.units = "days"
    itimeid.time_origin = "01-JAN-1900 00:00:00"

    z_tdim = ncfile.createDimension("zt",pyom.nz)
    z_udim = ncfile.createDimension("zw",pyom.nz)
    z_tid = ncfile.createVariable("zt","f8",("zt",))
    z_uid = ncfile.createVariable("zw","f8",("zw",))
    z_tid.long_name = "Vertical coordinate on T grid"
    z_tid.units = "m"
    z_uid.long_name = "Vertical coordinate on W grid"
    z_uid.units = "m"
    z_tid[...] = pyom.zt
    z_uid[...] = pyom.zw

    if pyom.coord_degree:
        lon_tid[...] = pyom.xt[2:-2]
        lon_uid[...] = pyom.xu[2:-2]
        lat_tid[...] = pyom.yt[2:-2]
        lat_uid[...] = pyom.yu[2:-2]
    else:
        lon_tid[...] = pyom.xt[2:-2] / 1e3
        lon_uid[...] = pyom.xu[2:-2] / 1e3
        lat_tid[...] = pyom.yt[2:-2] / 1e3
        lat_uid[...] = pyom.yu[2:-2] / 1e3

    ht_id = ncfile.createVariable("ht","f8",("xt","yt"))
    ht_id.long_name = "Depth"
    ht_id.units = "m"
    ht_id[...] = pyom.ht[2:-2, 2:-2]

@pyom_method
def panic_snap(pyom):
    print("Writing snapshot before panic shutdown")
    if not pyom.enable_diag_snapshots:
        init_snap_cdf(pyom)
    diag_snap(pyom)


Var = namedtuple("variable", ["name","dims","units"])
T_GRID = ("xt","yt","zt","Time")
U_GRID = ("xu","yt","zt","Time")
V_GRID = ("xt","yu","zt","Time")
W_GRID = ("xt","yt","zw","Time")

MAIN_VARIABLES = {
    "temp": Var("Temperature",T_GRID, "deg C"),
    "salt": Var("Salinity",T_GRID, "g/kg"),
    "u": Var("Zonal velocity", U_GRID, "m/s"),
    "v": Var("Meridional velocity", V_GRID, "m/s"),
    "w": Var("Vertical velocity", W_GRID, "m/s"),
    "forc_temp_surface": Var("Surface temperature flux", ("xt","yt","Time"), "m^2/s^2"),
    "forc_salt_surface": Var("Surface salinity flux", ("xt","yt","Time"), "m^2/s^2"),
    "surface_taux": Var("Surface wind stress", ("xu","yt","Time"), "m^2/s^2"),
    "surface_tauy": Var("Surface wind stress", ("xt","yu","Time"), "m^2/s^2"),
    "Nsqr": Var("Square of stability frequency", W_GRID, "1/s^2"),
    "kappaH": Var("Vertical diffusivity", W_GRID, "m^2/s"),
}
CONDITIONAL_VARIABLES = {
    "enable_conserve_energy": {
        "K_diss_v": Var("Dissipation of kinetic Energy", W_GRID, "m^2/s^3"),
        "K_diss_bot": Var("Dissipation of kinetic Energy", W_GRID, "m^2/s^3"),
        "K_diss_h": Var("Dissipation of kinetic Energy", W_GRID, "m^2/s^3"),
        "P_diss_v": Var("Dissipation of potential Energy", W_GRID, "m^2/s^3"),
        "P_diss_nonlin": Var("Dissipation of potential Energy", W_GRID, "m^2/s^3"),
        "P_diss_iso": Var("Dissipation of potential Energy", W_GRID, "m^2/s^3"),
        "P_diss_skew": Var("Dissipation of potential Energy", W_GRID, "m^2/s^3"),
        "P_diss_hmix": Var("Dissipation of potential Energy", W_GRID, "m^2/s^3"),
        "K_diss_gm": Var("Dissipation of mean energy", W_GRID, "m^2/s^3")
    },
    "enable_streamfunction": {
        "psi": Var("Streamfunction", ("xu","yu","Time"), "m^3/s"),
        "psin": Var("Boundary streamfunction", ("xu","yu","Time","isle"), "m^3/s")
    },
    "not enable_streamfunction": {
        "surf_press": Var("Surface pressure", ("xt","yt","Time"), "m^2/s^2")
    },
    "not enable_hydrostatic": {
        "p_hydro": Var("Hydrostatic pressure", T_GRID, "m^2/s^2"),
        "p_non_hydro": Var("Non-hydrostatic pressure", T_GRID, "m^2/s^2")
    },
    "enable_skew_diffusion": {
        "B1_gm": Var("Zoncal component of GM streamfunction", V_GRID, "m^2/s"),
        "B2_gm": Var("Meridional component of GM streamfunction", U_GRID, "m^2/s")
    },
    "enable_TEM_friction": {
        "kappa_gm": Var("Vertical diffusivity", W_GRID, "m^2/s"),
    },
    "enable_tke": {
        "tke": Var("Turbulent kinetic energy", W_GRID, "m^2/s^2"),
        "Prandtl": Var("Prandtl number", W_GRID, ""),
        "mxl": Var("Mixing length", W_GRID, "m"),
        "tke_diss": Var("TKE dissipation", W_GRID, "m^2/s^3"),
        "forc_tke": Var("TKE surface flux", ("xt","yt","Time"), "m^3/s^3"),
        "tke_surf_corr": Var("Correction of TKE surface flux", ("xt","yt","Time"), "m^3/s^3"),
    },
    "enable_eke": {
        "eke": Var("meso-scale energy", W_GRID, "m^2/s^2"),
        "L_Rossby": Var("Rossby radius", ("xt","yt","Time"), "m"),
        "L_Rhines": Var("Rhines scale", ("xt","yt","Time"), "m"),
        "K_gm": Var("Skewness diffusivity", W_GRID, "m^2/s"),
        "eke_diss_iw": Var("Dissipation of EKE to IW", W_GRID, "m^2/s^3"),
        "eke_diss_tke": Var("Dissipation of EKE to TKE", W_GRID, "m^2/s^3"),
        "eke_bot_flux": Var("Flux by bottom friction", ("xt","yt","Time"), "m^3/s^3"),
    },
    "enable_eke_leewave_dissipation": {
        "c_lee": Var("Lee wave dissipation coefficient", ("xt","yt","Time"), "1/s"),
        "eke_lee_flux": Var("Lee wave flux", ("xt","yt","Time"), "m^3/s^3"),
        "c_Ri_diss": Var("Interior dissipation coefficient", W_GRID, "1/s"),
    },
    "enable_idemix": {
        "E_iw": Var("Internal wave energy", W_GRID, "m^2/s^2"),
        "c0": Var("Vertical IW group velocity", W_GRID, "m/s"),
        "v0": Var("Horizontal IW group velocity", W_GRID, "m/s"),
        "iw_diss": Var("IW dissipation", W_GRID, "m^2/s^3"),
        "forc_iw_surface": Var("IW surface forcing", ("xt","yt"), "m^3/s^3"),
        "forc_iw_bottom": Var("IW bottom forcing", ("xt","yt"), "m^3/s^3"),
    },
    "enable_idemix_M2": {
        "E_M2": Var("M2 energy", ("xt","yt","Time"), "m^3/s^2"),
        "E_struct_M2": Var("M2 structure function", T_GRID, ""),
        "cg_M2": Var("M2 group velocity", ("xt","yt","Time"), "m/s"),
        "kdot_x_M2": Var("M2 refraction", ("xu","yt","Time"), "1/s"),
        "kdot_y_M2": Var("M2 refraction", ("xt","yu","Time"), "1/s"),
        "tau_M2": Var("M2 decay time scale", ("xt","yt","Time"), "1/s"),
        "alpha_M2_cont": Var("M2-continuum coupling coefficient", ("xt","yt","Time"), "s/m^3"),
        "forc_M2": Var("M2 forcing", ("xt","yt","Time"), "m^3/s^3"),
    },
    "enable_idemix_niw": {
        "E_niw": Var("NIW energy", ("xt","yt","Time"), "m^3/s^2"),
        "E_struct_niw": Var("NIW structure function", T_GRID, ""),
        "cg_niw": Var("NIW group velocity", ("xt","yt","Time"), "m/s"),
        "kdot_x_niw": Var("NIW refraction", ("xu","yt","Time"), "1/s"),
        "kdot_y_niw": Var("NIW refraction", ("xt","yu","Time"), "1/s"),
        "tau_niw": Var("NIW decay time scale", ("xt","yt","Time"), "1/s"),
        "forc_niw": Var("NIW forcing", ("xt","yt","Time"), "m^3/s^3"),
    },
}
FILL_VALUE = -1e33

@pyom_method
def _init_var(pyom, key, var, ncfile):
    v = ncfile.createVariable(key, "f8", var.dims, fill_value=FILL_VALUE)
    v.long_name = var.name
    v.units = var.units
    v.missing_value = FILL_VALUE

@pyom_method
def _get_condition(pyom, condition):
    return not getattr(pyom,condition[4:]) if condition.startswith("not ") else getattr(pyom, condition)

@pyom_method
def init_snap_cdf(pyom):
    """
    initialize NetCDF snapshot file
    """
    print("Preparing file {}".format(pyom.snap_file))

    with threaded_netcdf(pyom, Dataset(pyom.snap_file, "w"), file_id="snapshot") as snap_dataset:
        def_grid_cdf(pyom, snap_dataset)
        for key, var in MAIN_VARIABLES.items():
            _init_var(pyom,key,var,snap_dataset)
        for condition, vardict in CONDITIONAL_VARIABLES.items():
            if _get_condition(pyom,condition):
                for key, var in vardict.items():
                    _init_var(pyom,key,var,snap_dataset)

@pyom_method
def _write_var(pyom, key, var, n, ncfile):
    var_data = getattr(pyom,key)
    if var_data.ndim == 4:
        var_data = var_data[..., pyom.tau]
    grid_masks = {T_GRID: pyom.maskT, U_GRID: pyom.maskU,
                  V_GRID: pyom.maskV, W_GRID: pyom.maskW}
    for grid, mask in grid_masks.items():
        if var.dims == grid:
            var_data = np.where(mask.astype(np.bool), var_data, FILL_VALUE)
    ncfile[key][..., n] = var_data[2:-2, 2:-2, ...]

@pyom_method
def diag_snap(pyom):
    time_in_days = pyom.itt * pyom.dt_tracer / 86400.
    if time_in_days < 1.0:
        print(" writing snapshot at {}s".format(time_in_days * 86400.))
    else:
        print(" writing snapshot at {}d".format(time_in_days))

    with threaded_netcdf(pyom, Dataset(pyom.snap_file, "a"), file_id="snapshot") as snap_dataset:
        snapshot_number = snap_dataset["Time"].size
        snap_dataset["Time"][snapshot_number] = time_in_days
        for key, var in MAIN_VARIABLES.items():
            _write_var(pyom, key, var, snapshot_number, snap_dataset)
        for condition, vardict in CONDITIONAL_VARIABLES.items():
            if _get_condition(pyom, condition):
                _write_var(pyom, key, var, snapshot_number, snap_dataset)
