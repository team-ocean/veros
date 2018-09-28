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
    ("kappaM_min", Setting(0., float, "")),
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
    ("light_attenuation_phytoplankton", Setting(0.047, float, "Light attenuation of phytoplankton")),
    ("light_attenuation_water", Setting(0.04, float, "Light attenuation of water")),
    ("rctheta", Setting(1.33, float, "Angle of incidence")), # TODO what angle?
    ("nud0", Setting(0.07 / 86400, float, "Remineralization rate of detritus [1/sec]")),
    ("bbio", Setting(1.066, float, "??????")),
    ("cbio", Setting(1.0, float, "???????")),
    ("abio_P", Setting(0.6 / 86400, float, "Maximum growth rate parameter in [1/sec]")),  # TODO of what?
    ("gbio", Setting(0.38 / 86400, float, "Maximum grazing rate at 0 deg C [1/day]")),
    ("nupt0", Setting(0.01 / 86400, float, "Fast-recycling mortality rate of phytoplankton [1/sec]")),
    ("saturation_constant_N", Setting(0.7, float, "Half saturation constant for N uptate [mmol N / m^3]")),
    ("saturation_constant_Z_grazing", Setting(0.15, float, "Half saturation constant for Z grazing [mmol/m^3]")),
    ("specific_mortality_phytoplankton", Setting(0.03 / 86400, float, "Specific mortality rate of phytoplankton")),
    ("quadric_mortality_zooplankton", Setting(0.06 / 86400**2, float, "Quadric mortality rate of zooplankton [???]")),
    ("assimilation_efficiency", Setting(0.70, float, "gamma1")),  # TODO comment
    ("zooplankton_growth_efficiency", Setting(0.6, float, "Zooplankton growth efficiency")),
    ("wd0", Setting(16. / 86400, float, "Sinking speed of detritus at surface [m/s]")),
    ("mwz", Setting(1000, float, "Depth below which sinking speed of detritus remains constant [m]")),
    ("mw", Setting(0.02 / 86400, float, "Increase in sinking speed with depth [1/sec]")),
    ("zprefP", Setting(.3, float, "Zooplankton preference for grazing on Phytoplankton")),
    ("zprefZ", Setting(.3, float, "Zooplankton preference for grazing on other zooplankton")),
    ("zprefDet", Setting(.3, float, "Zooplankton preference for grazing on detritus")),
    ("redfield_ratio_PN", Setting(1./16, float, "Refield ratio for N/P")),
    ("trcmin", Setting(0, float, "Minimum npzd tracer value")),

    # NPZD with N
    ("jdiar", Setting(0.08, float, "Factor reducing the growth rate of diazotrophs")),
    ("nudon0", Setting(2.33e-5 / 86400, float, "DON remineralization rate [1/sec]")),
    ("nudop0", Setting(7e-5 / 86400, float, "DOP remineralization rate [1/sec]")),
    ("nupt0_D", Setting(0.001 / 86400, float, "Fast-recycling mortality rate of diazotrophs [1/sec]")),
    ("hdop", Setting(0.4, float, "DOP growth rate handicap")),
    ("zprefD", Setting(.1, float, "Z preference for diazotrophs")),
    ("specific_mortality_diazotroph", Setting(0.0001 / 86400, float, "Specific mortality rate of diazotrophs [1/sec]")),
    ("diazotroph_NP", Setting(28, float, "Diazotroph N:P ratio")),
    ("dfr", Setting(0.08, float, "phtoplankton mortality refactory/semi-labile DOM fraction")),
    ("dfrt", Setting(0.08, float, "phtoplankton fast recycling refactory/semi-labile DOM fraction")),


])


def set_default_settings(vs):
    for key, setting in SETTINGS.items():
        setattr(vs, key, setting.type(setting.default))


def check_setting_conflicts(vs):
    if vs.enable_tke and not vs.enable_implicit_vert_friction:
        raise RuntimeError("use TKE model only with implicit vertical friction"
                           "(set enable_implicit_vert_fricton)")

    if vs.enable_npzd:
        if vs.dt_bio > vs.dt_mom:
            raise RuntimeError("Biological timestep must be smaller than or equal to momentum timestep (ensure dt_bio > dt_mom)")

        # if ((vs.dt_mom / vs.dt_bio) % 1 is not 0):
        #     raise RuntimeError("Momentum timestep must be divisible by biological timestep")

        if not round(vs.zprefP + vs.zprefZ + vs.zprefDet + vs.zprefD, 3) == 1:
            raise RuntimeError("Zooplankton preferences must add to 1 it was {}".format(vs.zprefP + vs.zprefZ + vs.zprefDet + vs.zprefD))
