from collections import namedtuple

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

Var = namedtuple("variable", ["name", "dims", "time_dependent", "units",
                             "long_description", "dtype", "output"])

XT = ("xt",)
XU = ("xu",)
YT = ("yt",)
YU = ("yu",)
ZT = ("zt",)
ZW = ("zw",)
T_HOR = ("xt","yt")
U_HOR = ("xu","yt")
V_HOR = ("xt","yu")
T_GRID = ("xt","yt","zt")
U_GRID = ("xu","yt","zt")
V_GRID = ("xt","yu","zt")
W_GRID = ("xt","yt","zw")
ZETA_GRID = ("xu","yu","zt")
TIMESTEPS = (3,)

MAIN_VARIABLES = {
    "zt": Var(
        "Vertical coordinate (T)", ZT, "m", "Vertical coordinate", "float", True
    ),
    "zw": Var(
        "Vertical coordinate (W)", ZW, "m", "Vertical coordinate", "float", True
    ),
    "dzt": Var(
        "Vertical spacing (T)", ZT, "m", "Vertical spacing", "float", True
    ),
    "dzw": Var(
        "Vertical spacing (W)", ZW, "m", "Vertical spacing", "float", True
    ),
    "cost": Var(
        "Metric factor (T)", YT, "1", "Metric factor for spherical coordinates",
        "float", False
    ),
    "cosu": Var(
        "Metric factor (U)", YU, "1", "Metric factor for spherical coordinates",
        "float", False
    ),
    "tantr": Var(
        "Metric factor", YT, "1", "Metric factor for spherical coordinates",
        "float", False
    ),
    "coriolis_t": Var(
        "Coriolis frequency", T_HOR, "1/s",
        "Coriolis frequency at T grid point", "float", False
    ),
    "coriolis_h": Var(
        "Horizontal Coriolis frequency", T_HOR, "1/s",
        "Horizontal Coriolis frequency at T grid point", "float", False
    ),

    "kbot": Var(
        "Index of deepest cell", T_HOR, "",
        "Index of the deepest grid cell (counting from 1, 0 means all land)",
        "int", False
    ),
    "ht": Var(
        "Total depth (T)", T_HOR, "m", "Total depth of the water column", "float", True
    ),
    "hu": Var(
        "Total depth (U)", U_HOR, "m", "Total depth of the water column", "float", True
    ),
    "hv": Var(
        "Total depth (V)", V_HOR, "m", "Total depth of the water column", "float", True
    ),
    "hur": Var(
        "Total depth (U), masked", U_HOR, "m",
        "Total depth of the water column (masked)", "float", False
    ),
    "hvr": Var(
        "Total depth (V), masked", V_HOR, "m",
        "Total depth of the water column (masked)", "float", False
    ),
    "beta": Var(
        "Change of Coriolis freq.", T_HOR, "1/(ms)",
        "Change of Coriolis frequency with latitude", "float", True
    ),
    "area_t": Var(
        "Area of T-box", T_HOR, "m^2", "Area of T-box", "float", True
    ),
    "area_u": Var(
        "Area of U-box", U_HOR, "m^2", "Area of U-box", "float", True
    ),
    "area_v": Var(
        "Area of V-box", V_HOR, "m^2", "Area of V-box", "float", True
    ),

    "maskT": Var(
        "Mask for tracer points", T_GRID, "",
        "Mask in physical space for tracer points", "bool", False
    ),
    "maskU": Var(
        "Mask for U points", U_GRID, "",
        "Mask in physical space for U points", "bool", False
    ),
    "maskV": Var(
        "Mask for V points", V_GRID, "",
        "Mask in physical space for V points", "bool", False
    ),
    "maskW": Var(
        "Mask for W points", W_GRID, "",
        "Mask in physical space for W points", "bool", False
    ),
    "maskZ": Var(
        "Mask for Zeta points", ZETA_GRID, "",
        "Mask in physical space for Zeta points", "bool", False
    ),

    "rho": Var(
        "Density", T_GRID + TIMESTEPS, "kg/m^3", "Potential density", "float", True
    ),
    "int_drhodT": Var(
        "Der. of dyn. enthalpy by temperature", T_GRID + TIMESTEPS, "?",
        "Partial derivative of dynamic enthalpy by temperature", "float", True
    ),
    "int_drhodS": Var(
        "Der. of dyn. enthalpy by salinity", T_GRID + TIMESTEPS, "?",
        "Partial derivative of dynamic enthalpy by salinity", "float", True
    ),
    "Nsqr": Var(
        "Square of stability frequency", W_GRID + TIMESTEPS, "1/s^2",
        "Square of stability frequency", "float", True
    ),
    "Hd": Var(
        "Dynamic enthalpy", T_GRID + TIMESTEPS, "?", "Dynamic enthalpy", "float", True
    ),
    "dHd": Var(
        "Change of dyn. enth. by adv.", T_GRID + TIMESTEPS, "?",
        "Change of dynamic enthalpy due to advection", "float", True
    ),

    "temp": Var(
        "Temperature", T_GRID + TIMESTEPS, "deg C",
        "Conservative temperature", "float", True
    ),
    "dtemp": Var(
        "Temperature tendency", T_GRID + TIMESTEPS, "deg C/s",
        "Conservative temperature tendency", "float", False
    ),
    "salt": Var(
        "Salinity", T_GRID + TIMESTEPS, "g/kg", "Salinity", "float", True
    ),
    "dsalt": Var(
        "Salinity tendency", T_GRID + TIMESTEPS, "g/(kg s)", "float", False
    ),
    "dtemp_vmix": Var(
        "Change of temp. by vertical mixing", T_GRID, "deg C/s", "float", False
    ),

    "u": Var("Zonal velocity", U_GRID, "m/s", ),
    "v": Var("Meridional velocity", V_GRID, "m/s"),
    "w": Var("Vertical velocity", W_GRID, "m/s"),
    "forc_temp_surface": Var("Surface temperature flux", ("xt","yt","Time"), "m^2/s^2"),
    "forc_salt_surface": Var("Surface salinity flux", ("xt","yt","Time"), "m^2/s^2"),
    "surface_taux": Var("Surface wind stress", ("xu","yt","Time"), "m^2/s^2"),
    "surface_tauy": Var("Surface wind stress", ("xt","yu","Time"), "m^2/s^2"),
    "kappaH": Var("Vertical diffusivity", W_GRID, "m^2/s"),
}

CONDITIONAL_VARIABLES = {
    "coord_degree": {
        "xt": Var(
            "Zonal coordinate (T)", XT, False, "deg longitude",
            "Zonal (x) coordinate of T-grid point",
            "float", True
        ),
        "xu": Var(
            "Zonal coordinate (U)", XU, False, "deg longitude",
            "Zonal (x) coordinate of U-grid point",
            "float", True
        ),
        "yt": Var(
            "Meridional coordinate (T)", YT, False, "deg latitude",
            "Meridional (y) coordinate of T-grid point",
            "float", True
        ),
        "yu": Var(
            "Meridional coordinate (U)", YU, False, "deg latitude",
            "Meridional (y) coordinate of U-grid point",
            "float", True
        ),
        "dxt": Var(
            "Zonal T-grid spacing", XT, False, "deg longitude",
            "Zonal (x) spacing of T-grid point",
            "float", True
        ),
        "dxu": Var(
            "Zonal U-grid spacing", XU, False, "deg longitude",
            "Zonal (x) spacing of U-grid point",
            "float", True
        ),
        "dyt": Var(
            "Meridional T-grid spacing", YT, False, "deg latitude",
            "Meridional (y) spacing of T-grid point",
            "float", True
        ),
        "dyu": Var(
            "Meridional U-grid spacing", YU, False, "deg latitude",
            "Meridional (y) spacing of U-grid point",
            "float", True
        )
    },
    "not coord_degree": {
        "xt": Var(
            "Zonal coordinate (T)", XT, False, "m",
            "Zonal (x) coordinate of T-grid point",
            "float", True
        ),
        "xu": Var(
            "Zonal coordinate (U)", XU, False, "m",
            "Zonal (x) coordinate of U-grid point",
            "float", True
        ),
        "yt": Var(
            "Meridional coordinate (T)", YT, False, "m",
            "Meridional (y) coordinate of T-grid point",
            "float", True
        ),
        "yu": Var(
            "Meridional coordinate (U)", YU, False, "m",
            "Meridional (y) coordinate of U-grid point",
            "float", True
        ),
        "dxt": Var(
            "Zonal T-grid spacing", XT, False, "m",
            "Zonal (x) spacing of T-grid point",
            "float", True
        ),
        "dxu": Var(
            "Zonal U-grid spacing", XU, False, "m",
            "Zonal (x) spacing of U-grid point",
            "float", True
        ),
        "dyt": Var(
            "Meridional T-grid spacing", YT, False, "m",
            "Meridional (y) spacing of T-grid point",
            "float", True
        ),
        "dyu": Var(
            "Meridional U-grid spacing", YU, False, "m",
            "Meridional (y) spacing of U-grid point",
            "float", True
        )
    },
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
