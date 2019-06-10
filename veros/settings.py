from collections import namedtuple, OrderedDict

Setting = namedtuple("setting", ("default", "type", "description"))

SETTINGS = OrderedDict([
    ("identifier", Setting("UNNAMED", str, "Identifier of the current simulation")),

    # Model parameters
    ("nx", Setting(0, int, "Grid points in zonal (x) direction")),
    ("ny", Setting(0, int, "Grid points in meridional (y,j) direction")),
    ("nz", Setting(0, int, "Grid points in vertical (z,k) direction")),
    ("dt_mom", Setting(0., float, "Time step in seconds for momentum")),
    ("dt_tracer", Setting(0., float, "Time step for tracers, can be larger than dt_mom")),
    ("dt_tke", Setting(0., float, "Time step for TKE module, currently set to dt_mom (unused)")),
    ("dt_bio", Setting(0, float, "Time step for npzd, must be smaller than dt_mom")),
    ("runlen", Setting(0., float, "Length of simulation in seconds")),
    ("AB_eps", Setting(0.1, float, "Deviation from Adam-Bashforth weighting")),

    # Logical switches for general model setup
    ("coord_degree", Setting(False, bool, "either spherical (True) or cartesian (False) coordinates")),
    ("enable_cyclic_x", Setting(False, bool, "enable cyclic boundary conditions")),
    ("eq_of_state_type", Setting(1, int, "equation of state: 1: linear, 3: nonlinear with comp., 5: TEOS")),
    ("enable_implicit_vert_friction", Setting(False, bool, "enable implicit vertical friction")),
    ("enable_explicit_vert_friction", Setting(False, bool, "enable explicit vertical friction")),
    ("enable_hor_friction", Setting(False, bool, "enable horizontal friction")),
    ("enable_hor_diffusion", Setting(False, bool, "enable horizontal diffusion")),
    ("enable_biharmonic_friction", Setting(False, bool, "enable biharmonic horizontal friction")),
    ("enable_biharmonic_mixing", Setting(False, bool, "enable biharmonic horizontal mixing")),
    ("enable_hor_friction_cos_scaling", Setting(False, bool, "scaling of hor. viscosity with cos(latitude)**cosPower")),
    ("enable_ray_friction", Setting(False, bool, "enable Rayleigh damping")),
    ("enable_bottom_friction", Setting(False, bool, "enable bottom friction")),
    ("enable_bottom_friction_var", Setting(False, bool, "enable bottom friction with lateral variations")),
    ("enable_quadratic_bottom_friction", Setting(False, bool, "enable quadratic bottom friction")),
    ("enable_tempsalt_sources", Setting(False, bool, "enable restoring zones, etc")),
    ("enable_momentum_sources", Setting(False, bool, "enable restoring zones, etc")),
    ("enable_superbee_advection", Setting(False, bool, "enable advection scheme with implicit mixing")),
    ("enable_conserve_energy", Setting(True, bool, "exchange energy consistently")),
    ("enable_store_bottom_friction_tke", Setting(False, bool, "transfer dissipated energy by bottom/rayleig fric. to TKE, else transfer to internal waves")),
    ("enable_store_cabbeling_heat", Setting(False, bool, "transfer non-linear mixing terms to potential enthalpy, else transfer to TKE and EKE")),
    ("enable_noslip_lateral", Setting(False, bool, "enable lateral no-slip boundary conditions in harmonic- and biharmonic friction.")),

    # External mode
    ("congr_epsilon", Setting(1e-12, float, "convergence criteria for Poisson solver")),
    ("congr_max_iterations", Setting(1000, int, "maximum number of Poisson solver iterations")),

    # Mixing parameter
    ("A_h", Setting(0.0, float, "lateral viscosity in m^2/s")),
    ("K_h", Setting(0.0, float, "lateral diffusivity in m^2/s")),
    ("r_ray", Setting(0.0, float, "Rayleigh damping coefficient in 1/s")),
    ("r_bot", Setting(0.0, float, "bottom friction coefficient in 1/s")),
    ("r_quad_bot", Setting(0.0, float, "qudratic bottom friction coefficient")),
    ("hor_friction_cosPower", Setting(3, float, "")),
    ("A_hbi", Setting(0.0, float, "lateral biharmonic viscosity in m^4/s")),
    ("K_hbi", Setting(0.0, float, "lateral biharmonic diffusivity in m^4/s")),
    ("kappaH_0", Setting(0.0, float, "")),
    ("kappaM_0", Setting(0.0, float, "fixed values for vertical viscosity/diffusivity which are set for no TKE model")),

    # Options for isopycnal mixing
    ("enable_neutral_diffusion", Setting(False, bool, "enable isopycnal mixing")),
    ("enable_skew_diffusion", Setting(False, bool, "enable skew diffusion approach for eddy-driven velocities")),
    ("enable_TEM_friction", Setting(False, bool, "TEM approach for eddy-driven velocities")),
    ("K_iso_0", Setting(0.0, float, "constant for isopycnal diffusivity in m^2/s")),
    ("K_iso_steep", Setting(0.0, float, "lateral diffusivity for steep slopes in m^2/s")),
    ("K_gm_0", Setting(0.0, float, "fixed value for K_gm which is set for no EKE model")),
    ("iso_dslope", Setting(0.0008, float, "parameters controlling max allowed isopycnal slopes")),
    ("iso_slopec", Setting(0.001, float, "parameters controlling max allowed isopycnal slopes")),

    # Idemix 1.0
    ("enable_idemix", Setting(False, bool, "")),
    ("tau_v", Setting(1.0 * 86400.0, float, "time scale for vertical symmetrisation")),
    ("tau_h", Setting(15.0 * 86400.0, float, "time scale for horizontal symmetrisation")),
    ("gamma", Setting(1.57, float, "")),
    ("jstar", Setting(10.0, float, "spectral bandwidth in modes")),
    ("mu0", Setting(4.0 / 3.0, float, "dissipation parameter")),
    ("enable_idemix_hor_diffusion", Setting(False, bool, "")),
    ("enable_eke_diss_bottom", Setting(False, bool, "")),
    ("enable_eke_diss_surfbot", Setting(False, bool, "")),
    ("eke_diss_surfbot_frac", Setting(1.0, float, "fraction which goes into bottom")),
    ("enable_idemix_superbee_advection", Setting(False, bool, "")),
    ("enable_idemix_upwind_advection", Setting(False, bool, "")),

    # TKE
    ("enable_tke", Setting(False, bool, "")),
    ("c_k", Setting(0.1, float, "")),
    ("c_eps", Setting(0.7, float, "")),
    ("alpha_tke", Setting(1.0, float, "")),
    ("mxl_min", Setting(1e-12, float, "")),
    ("kappaM_min", Setting(2e-4, float, "")),
    ("kappaM_max", Setting(100., float, "")),
    ("tke_mxl_choice", Setting(1, int, "")),
    ("enable_tke_superbee_advection", Setting(False, bool, "")),
    ("enable_tke_upwind_advection", Setting(False, bool, "")),
    ("enable_tke_hor_diffusion", Setting(False, bool, "")),
    ("K_h_tke", Setting(2000., float, "lateral diffusivity for tke")),

    # EKE
    ("enable_eke", Setting(False, bool, "")),
    ("eke_lmin", Setting(100.0, float, "minimal length scale in m")),
    ("eke_c_k", Setting(1.0, float, "")),
    ("eke_cross", Setting(1.0, float, "Parameter for EKE model")),
    ("eke_crhin", Setting(1.0, float, "Parameter for EKE model")),
    ("eke_c_eps", Setting(1.0, float, "Parameter for EKE model")),
    ("eke_k_max", Setting(1e4, float, "maximum of K_gm")),
    ("alpha_eke", Setting(1.0, float, "factor vertical friction")),
    ("enable_eke_superbee_advection", Setting(False, bool, "")),
    ("enable_eke_upwind_advection", Setting(False, bool, "")),
    ("enable_eke_isopycnal_diffusion", Setting(False, bool, "use K_gm also for isopycnal diffusivity")),

    ("enable_eke_leewave_dissipation", Setting(False, bool, "")),
    ("c_lee0", Setting(1., float, "")),
    ("eke_Ri0", Setting(200., float, "")),
    ("eke_Ri1", Setting(50., float, "")),
    ("eke_int_diss0", Setting(1. / (20 * 86400.), float, "")),
    ("kappa_EKE0", Setting(0.1, float, "")),
    ("eke_r_bot", Setting(0.0, float, "bottom friction coefficient")),
    ("eke_hrms_k0_min", Setting(0.0, float, "min value for bottom roughness parameter")),

    # New
    ("kappaH_min", Setting(0., float, "minimum value for vertical diffusivity")),
    ("enable_Prandtl_tke", Setting(True, bool, "Compute Prandtl number from stratification levels in TKE routine")),
    ("Prandtl_tke0", Setting(10., float, "Constant Prandtl number when stratification is neglected for kappaH computation in TKE routine")),
    ("verbose_island_routines", Setting(False, bool, "Print extra debugging output in island / boundary integral routines")),
    ("use_io_threads", Setting(True, bool, "Start extra threads for disk writes")),
    ("io_timeout", Setting(20, float, "Timeout in seconds while waiting for IO locks to be released")),
    ("enable_netcdf_zlib_compression", Setting(True, bool, "Use netCDF4's native zlib interface, which leads to smaller output files (but carries some computational overhead).")),
    ("enable_hdf5_gzip_compression", Setting(True, bool, "Use h5py's native gzip interface, which leads to smaller restart files (but carries some computational overhead).")),
    ("restart_input_filename", Setting("", str, "File name of restart input. If not given, no restart data will be read.")),
    ("restart_output_filename", Setting("{identifier}_{itt:0>4d}.restart.h5", str, "File name of restart output. May contain Python format syntax that is substituted with Veros attributes.")),
    ("restart_frequency", Setting(0, float, "Frequency (in seconds) to write restart data")),
    ("force_overwrite", Setting(False, bool, "Overwrite existing output files")),
    ("pyom_compatibility_mode", Setting(False, bool, "Force compatibility to pyOM2 (even reproducing bugs and other quirks). For testing purposes only.")),
    ("diskless_mode", Setting(False, bool, "Suppress all output to disk. Mainly used for testing purposes.")),
    ("default_float_type", Setting("float64", str, "Default type to use for floating point arrays (e.g. ``float32`` or ``float64``).")),
    ("use_amg_preconditioner", Setting(True, bool, "Use AMG preconditioner in Poisson solver if pyamg is installed.")),

    # NPZD
    ("enable_npzd", Setting(False, bool, "")),

    # I don't like keeping these in settings, but I can't make them in variables
    # maybe keep selected rules
    ("recycled", Setting({}, dict, "Amount of recycled material [mmol/m^3] for NPZD tracers")),
    ("mortality", Setting({}, dict, "Amount of dead plankton [mmol/m^3] by species")),
    ("net_primary_production", Setting({}, dict, "Primary production for each producing plankton species")),
    ("plankton_growth_functions", Setting({}, dict, "Collection of functions calculating growth for plankton by species")),
    ("limiting_functions", Setting({}, dict, "Collection of functions calculating limits to growth for plankton by species")),
    ("sinking_speeds", Setting({}, dict, "Speed with which the named tracers sink")),
    ("recycling_rates", Setting({}, dict, "Recycling rates for npzd tracers")),
    ("mortality_rates", Setting({}, dict, "Mortality rates for npzd tracers")),
    ("npzd_tracers", Setting({}, dict, "Dictionary whose values point to veros variables for npzd tracers")),
    ("npzd_rules", Setting([], list, "List of active rules in primary loop of BGC")),
    ("npzd_pre_rules", Setting([], list, "List of rules to executed in the pre loop of BGC")),
    ("npzd_post_rules", Setting([], list,  "Rules to be executed after primary bio loop")),
    ("npzd_available_rules", Setting({}, dict, "Every rule created is stored here, can be individual rules or collections of rules")),
    ("npzd_selected_rule_names", Setting([], list, "name of selected rules")),
    ("npzd_export", Setting({}, dict, "Exported material from npzd tracers by sinking")),
    ("npzd_import", Setting({}, dict, "Imported material from npzd tracers from layer above. Takes same value as npzd_export scaled by level differences. Sea surface is 0")),
    ("zprefs", Setting({}, dict, "Preference for zooplankton to graze on named tracers")),
    # TODO remove this, when merging npzd_objects
    ("npzd_transported_tracers", Setting([], list, "List of NPZD tracers which are transported")),
    ("npzd_advection_derivatives", Setting({}, dict, "Stores derivates of advection term for tracers")),
    ("temporary_tracers", Setting({}, dict, "Temporary copy of npzd_tracers for biogeochemistry loop")),





    ("light_attenuation_phytoplankton", Setting(0.047, float, "Light attenuation of phytoplankton")),
    ("light_attenuation_water", Setting(0.04, float, "Light attenuation of water [1/m]")),
    ("light_attenuation_ice", Setting(5.0, float, "Light attenuation of ice [1/m]")),
    # ("nud0", Setting(0.07 / 86400, float, "Remineralization rate of detritus [1/sec]")),
    # ("nud0", Setting(0, float, "Remineralization rate of detritus [1/sec]")),
    ("remineralization_rate_detritus", Setting(0, float, "Remineralization rate of detritus [1/sec]")),
    # ("bbio", Setting(1.066, float, "the b in b ** (c*T)")),
    ("bbio", Setting(0, float, "the b in b ** (c*T)")),
    # ("cbio", Setting(1.0, float, "the c in b ** (c*T)")),
    ("cbio", Setting(0, float, "the c in b ** (c*T)")),
    # ("abio_P", Setting(0.6 / 86400, float, "Maximum growth rate parameter for phytoplankton in [1/sec]")),
    # ("abio_P", Setting(0.0, float, "Maximum growth rate parameter for phytoplankton in [1/sec]")),
    ("maximum_growth_rate_phyto", Setting(0.0, float, "Maximum growth rate parameter for phytoplankton in [1/sec]")),
    # ("gbio", Setting(0.38 / 86400, float, "Maximum grazing rate at 0 deg C [1/sec]")),
    # ("gbio", Setting(0, float, "Maximum grazing rate at 0 deg C [1/sec]")),
    ("maximum_grazing_rate", Setting(0, float, "Maximum grazing rate at 0 deg C [1/sec]")),
    # ("nupt0", Setting(0.015 / 86400, float, "Fast-recycling mortality rate of phytoplankton [1/sec]")),
    # ("nupt0", Setting(0, float, "Fast-recycling mortality rate of phytoplankton [1/sec]")),
    ("fast_recycling_rate_phytoplankton", Setting(0, float, "Fast-recycling mortality rate of phytoplankton [1/sec]")),
    ("saturation_constant_N", Setting(0.7, float, "Half saturation constant for N uptate [mmol N / m^3]")),
    ("saturation_constant_Z_grazing", Setting(0.15, float, "Half saturation constant for Z grazing [mmol/m^3]")),
    # ("specific_mortality_phytoplankton", Setting(0.03 / 86400, float, "Specific mortality rate of phytoplankton")),
    ("specific_mortality_phytoplankton", Setting(0, float, "Specific mortality rate of phytoplankton")),
    # ("quadric_mortality_zooplankton", Setting(0.06 / 86400, float, "Quadric mortality rate of zooplankton [???]")),
    ("quadric_mortality_zooplankton", Setting(0, float, "Quadric mortality rate of zooplankton [1/ (mmol N ^2 s)]")),
    # ("assimilation_efficiency", Setting(0.70, float, "gamma1")),
    ("assimilation_efficiency", Setting(0, float, "Effiency with which ingested prey is converted growth in zooplankton, range: [0,1]")),
    # ("zooplankton_growth_efficiency", Setting(0.6, float, "Zooplankton growth efficiency")),
    ("zooplankton_growth_efficiency", Setting(0, float, "Zooplankton growth efficiency, range: [0,1]")),
    # ("wd0", Setting(5. / 86400, float, "Sinking speed of detritus at surface [m/s]")),
    ("wd0", Setting(0 / 86400, float, "Sinking speed of detritus at surface [m/s]")),
    ("mwz", Setting(1000, float, "Depth below which sinking speed of detritus remains constant [m]")),
    ("mw", Setting(0.02 / 86400, float, "Increase in sinking speed with depth [1/sec]")),
    ("zprefP", Setting(1, float, "Zooplankton preference for grazing on Phytoplankton")),
    ("zprefZ", Setting(1, float, "Zooplankton preference for grazing on other zooplankton")),
    ("zprefDet", Setting(1, float, "Zooplankton preference for grazing on detritus")),
    ("redfield_ratio_PN", Setting(1./16, float, "Refield ratio for P/N")),
    ("redfield_ratio_CP", Setting(7.1 * 16, float, "Refield ratio for C/P")),
    ("redfield_ratio_ON", Setting(10.6, float, "Redfield ratio for O/N")),
    ("redfield_ratio_CN", Setting(7.1, float, "Redfield ratio for C/N")),
    ("trcmin", Setting(1e-13, float, "Minimum npzd tracer value")),
    ("u1_min", Setting(1e-6, float, "Minimum u1 value for calculating avg J")),
    ("zooplankton_max_growth_temp", Setting(20.0, float, "Temperature (C) for which zooplankton growth rate no longer grows with temperature")),

    # NPZD with N
    # ("enable_nitrogen", Setting(False, bool, "")),
    # ("gd_min_diaz", Setting(1e-14, float, "Minimum value for gd for diazotroph growth")),
    # ("jdiar", Setting(0.08, float, "Factor reducing the growth rate of diazotrophs")),
    # ("nudon0", Setting(2.33e-5 / 86400, float, "DON remineralization rate [1/sec]")),
    # ("nudop0", Setting(7e-5 / 86400, float, "DOP remineralization rate [1/sec]")),
    # ("nupt0_D", Setting(0.001 / 86400, float, "Fast-recycling mortality rate of diazotrophs [1/sec]")),
    # ("hdop", Setting(0.4, float, "DOP growth rate handicap")),
    # ("zprefD", Setting(1.0 / 3, float, "Z preference for diazotrophs")),
    # ("specific_mortality_diazotroph", Setting(0.0001 / 86400, float, "Specific mortality rate of diazotrophs [1/sec]")),
    # ("diazotroph_NP", Setting(28, float, "Diazotroph N:P ratio")),
    # ("dfr", Setting(0.08, float, "phtoplankton mortality refactory/semi-labile DOM fraction")),
    # ("dfrt", Setting(0.08, float, "phtoplankton fast recycling refactory/semi-labile DOM fraction")),
    # ("bct_min_diaz", Setting(2.6, float, "Minmum value for b*c^T for calculating diazotroph growth")),

    # NPZD with caco3
    # ("enable_calcifiers", Setting(False, bool, "")),
    # ("light_attenuation_caco3", Setting(0.047, float, "Calcite light attenuation [1/(m * mmol/m^3)")),
    # ("zprefC", Setting(1, float, "Zooplankton preference for coccoliths")),
    # ("alpha_C", Setting(0.06 / 86400, float, "Initial slope P-I curve [(W/m^2)^-1/sec]")),
    # ("abio_C", Setting(0.52 / 86400, float, "a; Maximum growth rate parameter coccolithophore [1/sec]")),
    # ("specific_mortality_coccolitophore", Setting(0.03 / 86400, float, "Specific mortality rate coccolithophores [1/sec]")),
    # ("nuct0", Setting(0.015 / 86400, float, "Fast recycling rate? coccolithophores [1/sec]")),
    # ("wc0", Setting(35.0 / 86400, float, "Constant calcite sinking speed")),
    # ("mw_c", Setting(0.06 / 86400, float, "Calcite sinking speed increase with depth")),
    # ("dissk0", Setting(0.013 / 86400, float, "Initial dissolution rate parameter [1/sec]")),
    # ("Omega_c", Setting(1, float, "?????????????+ Never set or explained????????????????????")),
    # ("saturation_constant_NC", Setting(0.7, float, "Half saturation constant for N uptake for coccolitiophores [mmol N / m^3]")),
    ("capr", Setting(0.022, float, "Carbonate to carbon production ratio")),
    # # Ballast
    # ("bapr", Setting(0.05, float, "Detritus to carbonate ratio [mg POC/mg PIC]")),
    # # ("dcaco3", Setting(6500.0, float, "Calcite remineralization depth [m]")),
    # ("dcaco3", Setting(3500.0, float, "Calcite remineralization depth [m]")),

    # NPZD with iron
    # ("enable_iron", Setting(False, bool, "")),
    # ("pmax_P", Setting(0.15, float, "Phytoplankton biomass above which kfe increases [mmol N / m^3]")),
    # ("kfemin", Setting(0.04e-3, float, "Minimum half saturation constant for Fe limitation phyto [mmol Fe / m^3]")),
    # ("kfemax", Setting(0.2e-3, float, "Maximum half saturation constant for Fe limitation phyto [mmol Fe / m^3]")),
    # ("kfe_D", Setting(0.1e-3, float, "Half saturation constant diazotroph Fe limitation [mmol Fe / m^3]")),
    # ("kfemin_C", Setting(0.04e-3, float, "Half saturation constant diazotroph Fe limitation cocc [mmol Fe / m^3]")),
    # ("kfemax_C", Setting(0.4e-3, float, "Half saturation constant diazotroph Fe limitation cocc [mmol Fe / m^3]")),
    # ("pmax_C", Setting(0.15, float, "Biomass above which kfe increases [mmol N / m^3]")),
    # ("kfeleq", Setting(10.0 ** 5.5, float, "Fe-ligand stability constant [m^3/(mmol ligand)]")),
    # ("lig", Setting(1.0e-3, float, "Ligand concentration")),
    # ("thetamaxhi", Setting(0.04, float, "Maximum Chl:C ratio, abundant iron [gChl/(gC)]")),
    # ("thetamaxlo", Setting(0.01, float, "Maximum Chl:C ratio, extreme iron limitation [gChl/(gC)]")),
    # ("alphamax", Setting(73.6 * 1e-6 * 86400, float, "Maximum intial slope in PI-curve [??]")),
    # ("alphamin", Setting(18.4 * 1e-6 * 86400, float, "Minimum intial slope in PI-curve [??]")),
    # ("mc", Setting(12.011, float, "Molar mass of carbon")),
    # ("fetopsed", Setting(0.01, float, "Fe:P for sedimentary iron source [mmolFe/molP]")),
    # ("o2min", Setting(5.0, float, "Minimum O2 concentration for aerobic respiration [mmol/m^3]")),
    # ("kfeorg", Setting(0.45 / 86400, float, "Organic-matter dependent scavenging rate [(m^3/(gC s))^0.58]")),
    # ("rfeton", Setting(1e-6 * 6.625, float, "Uptake ratio of iron to nitrogen [mol Fe/mol N] = 10 micromol Fe / mol C")),
    # ("kfecol", Setting(0.005 / 86400, float, "Colloidal production and precipitation rate [s^-1]")),
])


def set_default_settings(vs):
    for key, setting in SETTINGS.items():
        setattr(vs, key, setting.type(setting.default))


def check_setting_conflicts(vs):
    if vs.enable_tke and not vs.enable_implicit_vert_friction:
        raise RuntimeError("use TKE model only with implicit vertical friction"
                           "(set enable_implicit_vert_fricton)")

    if vs.enable_npzd:
        if vs.dt_bio > vs.dt_tracer:
            raise RuntimeError("Biological timestep must be smaller than or equal to tracer timestep (ensure dt_bio > dt_tracer)")

        if ((vs.dt_tracer / vs.dt_bio) % 1 != 0.0):
            raise RuntimeError("Tracer timestep must be divisible by biological timestep, ratio was", vs.dt_tracer / vs.dt_bio)
