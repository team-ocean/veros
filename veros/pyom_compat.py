import os
from contextlib import contextmanager
from collections import defaultdict
import importlib.util

from veros import logger, runtime_settings, runtime_state, timer
from veros.state import get_default_state, resize_dimension
from veros.variables import get_shape


# all variables that are re-named or unique to Veros
VEROS_TO_PYOM_VAR = dict(
    # do not exist in pyom
    time=None,
    prho=None,
    land_map=None,
    isle=None,
    boundary_mask=None,
    line_dir_south_mask=None,
    line_dir_east_mask=None,
    line_dir_north_mask=None,
    line_dir_west_mask=None,
)

# all setting that are re-named or unique to Veros
VEROS_TO_PYOM_SETTING = dict(
    # do not exist in pyom
    identifier=None,
    enable_noslip_lateral=None,
    restart_input_filename=None,
    restart_output_filename=None,
    restart_frequency=None,
    kappaH_min=None,
    enable_kappaH_profile=None,
    enable_Prandtl_tke=None,
    Prandtl_tke0=None,
    biharmonic_friction_cosPower=None,
    # constants
    pi=None,
    radius=None,
    degtom=None,
    mtodeg=None,
    omega=None,
    rho_0=None,
    grav=None,
)


STREAMFUNCTION_VARS = ("psin", "dpsin", "line_psin")


def _load_fortran_module(module, path):
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pyom(pyom_lib):
    try:
        pyom_obj = _load_fortran_module("pyOM_code_MPI", pyom_lib)
        has_mpi = True
    except ImportError:
        pyom_obj = _load_fortran_module("pyOM_code", pyom_lib)
        has_mpi = False

    if runtime_state.proc_num > 1 and not has_mpi:
        raise RuntimeError("Given PyOM2 library was not built with MPI support")

    return pyom_obj


@contextmanager
def suppress_stdout(stdout_fd=1):
    old_stdout = os.dup(stdout_fd)

    with open(os.devnull, "wb") as void:
        os.dup2(void.fileno(), stdout_fd)

    try:
        yield
    finally:
        with os.fdopen(old_stdout, "wb") as std:
            os.dup2(std.fileno(), stdout_fd)


def pyom_from_state(state, pyom_obj, ignore_attrs=None, init_streamfunction=True):
    """Force-updates internal PyOM library state to match given Veros state."""
    if ignore_attrs is None:
        ignore_attrs = []

    pyom_modules = (
        pyom_obj.main_module,
        pyom_obj.isoneutral_module,
        pyom_obj.idemix_module,
        pyom_obj.tke_module,
        pyom_obj.eke_module,
    )

    def set_fortran_attr(attr, val):
        # fortran interface is all lower-case
        attr = attr.lower()

        for module in pyom_modules:
            if hasattr(module, attr):
                setattr(module, attr, val)
                break
        else:
            raise RuntimeError(f"Could not set attribute {attr} on Fortran library")

    # settings
    for setting, val in state.settings.items():
        setting = VEROS_TO_PYOM_SETTING.get(setting, setting)
        if setting is None or setting in ignore_attrs:
            continue

        set_fortran_attr(setting, val)

    _override_settings(pyom_obj)

    # allocate variables
    if runtime_state.proc_num > 1:
        pyom_obj.my_mpi_init(runtime_settings.mpi_comm.py2f())
    else:
        pyom_obj.my_mpi_init(0)

    pyom_obj.pe_decomposition()
    pyom_obj.allocate_main_module()
    pyom_obj.allocate_isoneutral_module()
    pyom_obj.allocate_tke_module()
    pyom_obj.allocate_eke_module()
    pyom_obj.allocate_idemix_module()

    # set variables
    for var, val in state.variables.items():
        var = VEROS_TO_PYOM_VAR.get(var, var)
        if var is None or var in ignore_attrs:
            continue

        if var in STREAMFUNCTION_VARS:
            continue

        set_fortran_attr(var, val)

    if init_streamfunction:
        with suppress_stdout():
            pyom_obj.streamfunction_init()

        for var in STREAMFUNCTION_VARS:
            set_fortran_attr(var, state.variables.get(var))

    # correct for 1-based indexing
    pyom_obj.main_module.tau += 1
    pyom_obj.main_module.taup1 += 1
    pyom_obj.main_module.taum1 += 1

    # diagnostics
    diag_settings = (
        ("cfl_monitor", "output_frequency", "ts_monint"),
        ("tracer_monitor", "output_frequency", "trac_cont_int"),
        ("snapshot", "output_frequency", "snapint"),
        ("averages", "output_frequency", "aveint"),
        ("averages", "sampling_frequency", "avefreq"),
        ("overturning", "output_frequency", "overint"),
        ("overturning", "sampling_frequency", "overfreq"),
        ("energy", "output_frequency", "energint"),
        ("energy", "sampling_frequency", "energfreq"),
    )

    for diag, param, attr in diag_settings:
        if diag in state.diagnostics:
            set_fortran_attr(attr, getattr(diag, param))

    return pyom_obj


def _override_settings(pyom_obj):
    """Manually force some settings to ensure compatibility."""
    m = pyom_obj.main_module
    m.n_pes_i, m.n_pes_j = runtime_settings.num_proc

    # define processor boundary idx (1-based)
    ipx, ipy = runtime_state.proc_idx
    m.is_pe = (m.nx // m.n_pes_i) * ipx + 1
    m.ie_pe = (m.nx // m.n_pes_i) * (ipx + 1)
    m.js_pe = (m.ny // m.n_pes_j) * ipy + 1
    m.je_pe = (m.ny // m.n_pes_j) * (ipy + 1)

    # force settings that are not supported by Veros
    idm = pyom_obj.idemix_module
    eke = pyom_obj.eke_module

    m.enable_streamfunction = True
    m.enable_hydrostatic = True
    m.congr_epsilon = 1e-8
    m.congr_max_iterations = 10_000
    m.enable_congrad_verbose = False
    m.enable_free_surface = False
    eke.enable_eke_leewave_dissipation = False
    idm.enable_idemix_m2 = False
    idm.enable_idemix_niw = False

    return pyom_obj


def state_from_pyom(pyom_obj):
    from veros.core.operators import numpy as npx

    state = get_default_state()

    pyom_modules = (
        pyom_obj.main_module,
        pyom_obj.isoneutral_module,
        pyom_obj.idemix_module,
        pyom_obj.tke_module,
        pyom_obj.eke_module,
    )

    def get_fortran_attr(attr):
        # fortran interface is all lower-case
        attr = attr.lower()

        for module in pyom_modules:
            if hasattr(module, attr):
                return getattr(module, attr)
        else:
            raise RuntimeError(f"Could not get attribute {attr} from Fortran library")

    with state.settings.unlock():
        for setting in state.settings.fields():
            setting = VEROS_TO_PYOM_SETTING.get(setting, setting)
            if setting is None:
                continue

            state.settings.update({setting: get_fortran_attr(setting)})

    state.initialize_variables()
    resize_dimension(state, "isle", int(pyom_obj.main_module.nisle))

    with state.variables.unlock():
        state.variables.isle = npx.arange(state.dimensions["isle"])

        for var, val in state.variables.items():
            var = VEROS_TO_PYOM_VAR.get(var, var)
            if var is None:
                continue

            try:
                new_val = get_fortran_attr(var)
            except RuntimeError:
                continue

            if new_val is None:
                continue

            try:
                new_val = npx.broadcast_to(new_val, val.shape)
            except ValueError:
                raise ValueError(f"variable {var} has incompatible shapes: {val.shape}, {new_val.shape}")

            state.variables.update({var: new_val})

    return state


def setup_pyom(pyom_obj, set_parameter, set_grid, set_coriolis, set_topography, set_initial_conditions, set_forcing):
    if runtime_state.proc_num > 1:
        pyom_obj.my_mpi_init(runtime_settings.mpi_comm.py2f())
    else:
        pyom_obj.my_mpi_init(0)

    set_parameter(pyom_obj)

    pyom_obj.pe_decomposition()
    pyom_obj.allocate_main_module()
    pyom_obj.allocate_isoneutral_module()
    pyom_obj.allocate_tke_module()
    pyom_obj.allocate_eke_module()
    pyom_obj.allocate_idemix_module()

    set_grid(pyom_obj)
    pyom_obj.calc_grid()

    set_coriolis(pyom_obj)
    pyom_obj.calc_beta()

    set_topography(pyom_obj)
    pyom_obj.calc_topo()
    pyom_obj.calc_spectral_topo()

    set_initial_conditions(pyom_obj)
    pyom_obj.calc_initial_conditions()

    pyom_obj.streamfunction_init()

    set_forcing(pyom_obj)

    pyom_obj.check_isoneutral_slope_crit()


def run_pyom(pyom_obj, set_forcing, after_timestep=None):
    timers = defaultdict(timer.Timer)

    f = pyom_obj
    m = pyom_obj.main_module
    idm = pyom_obj.idemix_module
    ekm = pyom_obj.eke_module
    tkm = pyom_obj.tke_module

    logger.info(f"Starting integration for {float(m.runlen):.2e}s")

    m.time = 0.0
    while m.time < m.runlen:
        logger.info(f"Current iteration: {m.itt}")

        with timers["main"]:
            set_forcing(pyom_obj)

            if idm.enable_idemix:
                f.set_idemix_parameter()

            f.set_eke_diffusivities()
            f.set_tke_diffusivities()

            with timers["momentum"]:
                f.momentum()

            with timers["temperature"]:
                f.thermodynamics()

            if ekm.enable_eke or tkm.enable_tke or idm.enable_idemix:
                f.calculate_velocity_on_wgrid()

            with timers["eke"]:
                if ekm.enable_eke:
                    f.integrate_eke()

            with timers["idemix"]:
                if idm.enable_idemix:
                    f.integrate_idemix()

            with timers["tke"]:
                if tkm.enable_tke:
                    f.integrate_tke()

            """
            Main boundary exchange
            for density, temp and salt this is done in integrate_tempsalt.f90
            """
            f.border_exchg_xyz(
                m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx, m.je_pe + m.onx, m.u[:, :, :, m.taup1 - 1], m.nz
            )
            f.setcyclic_xyz(
                m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx, m.je_pe + m.onx, m.u[:, :, :, m.taup1 - 1], m.nz
            )
            f.border_exchg_xyz(
                m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx, m.je_pe + m.onx, m.v[:, :, :, m.taup1 - 1], m.nz
            )
            f.setcyclic_xyz(
                m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx, m.je_pe + m.onx, m.v[:, :, :, m.taup1 - 1], m.nz
            )

            if tkm.enable_tke:
                f.border_exchg_xyz(
                    m.is_pe - m.onx,
                    m.ie_pe + m.onx,
                    m.js_pe - m.onx,
                    m.je_pe + m.onx,
                    tkm.tke[:, :, :, m.taup1 - 1],
                    m.nz,
                )
                f.setcyclic_xyz(
                    m.is_pe - m.onx,
                    m.ie_pe + m.onx,
                    m.js_pe - m.onx,
                    m.je_pe + m.onx,
                    tkm.tke[:, :, :, m.taup1 - 1],
                    m.nz,
                )
            if ekm.enable_eke:
                f.border_exchg_xyz(
                    m.is_pe - m.onx,
                    m.ie_pe + m.onx,
                    m.js_pe - m.onx,
                    m.je_pe + m.onx,
                    ekm.eke[:, :, :, m.taup1 - 1],
                    m.nz,
                )
                f.setcyclic_xyz(
                    m.is_pe - m.onx,
                    m.ie_pe + m.onx,
                    m.js_pe - m.onx,
                    m.je_pe + m.onx,
                    ekm.eke[:, :, :, m.taup1 - 1],
                    m.nz,
                )
            if idm.enable_idemix:
                f.border_exchg_xyz(
                    m.is_pe - m.onx,
                    m.ie_pe + m.onx,
                    m.js_pe - m.onx,
                    m.je_pe + m.onx,
                    idm.e_iw[:, :, :, m.taup1 - 1],
                    m.nz,
                )
                f.setcyclic_xyz(
                    m.is_pe - m.onx,
                    m.ie_pe + m.onx,
                    m.js_pe - m.onx,
                    m.je_pe + m.onx,
                    idm.e_iw[:, :, :, m.taup1 - 1],
                    m.nz,
                )

            # diagnose vertical velocity at taup1
            f.vertical_velocity()

            # diagnose isoneutral streamfunction regardless of output settings
            f.isoneutral_diag_streamfunction()

        # shift time
        m.itt += 1
        m.time += m.dt_tracer

        if callable(after_timestep):
            after_timestep(pyom_obj)

        orig_taum1 = int(m.taum1)
        m.taum1 = m.tau
        m.tau = m.taup1
        m.taup1 = orig_taum1

        # NOTE: benchmarks parse this, do not change / remove
        logger.debug("Time step took {}s", timers["main"].last_time)

    logger.debug("Timing summary:")
    logger.debug(" setup time summary       = {}s", timers["setup"].total_time)
    logger.debug(" main loop time summary   = {}s", timers["main"].total_time)
    logger.debug("     momentum             = {}s", timers["momentum"].total_time)
    logger.debug("     thermodynamics       = {}s", timers["temperature"].total_time)
    logger.debug("     EKE                  = {}s", timers["eke"].total_time)
    logger.debug("     IDEMIX               = {}s", timers["idemix"].total_time)
    logger.debug("     TKE                  = {}s", timers["tke"].total_time)


def _generate_random_var(state, var):
    import numpy as onp

    meta = state.var_meta[var]
    shape = get_shape(state.dimensions, meta.dims)
    global_shape = get_shape(state.dimensions, meta.dims, local=False)

    if var == "kbot":
        val = onp.zeros(shape)
        val[2:-2, 2:-2] = onp.random.randint(1, state.dimensions["zt"], size=(shape[0] - 4, shape[1] - 4))
        island_mask = onp.random.choice(val[3:-3, 3:-3].size, size=10)
        val[3:-3, 3:-3].flat[island_mask] = 0
        return val

    if var in ("dxt", "dxu", "dyt", "dyu"):
        if state.settings.coord_degree:
            val = 80 / global_shape[0] * (1 + 1e-2 * onp.random.randn(*shape))
        else:
            val = 10_000e3 / global_shape[0] * (1 + 1e-2 * onp.random.randn(*shape))
        return val

    if var in ("dzt", "dzw"):
        val = 6000 / global_shape[0] * (1 + 1e-2 * onp.random.randn(*shape))
        return val

    if onp.issubdtype(onp.dtype(meta.dtype), onp.floating):
        val = onp.random.randn(*shape)
        if var in ("salt",):
            val = 35 + val

        return val

    if onp.issubdtype(onp.dtype(meta.dtype), onp.integer):
        val = onp.random.randint(0, 100, size=shape)
        return val

    if onp.issubdtype(onp.dtype(meta.dtype), onp.bool_):
        return onp.random.randint(0, 1, size=shape, dtype="bool")

    raise TypeError(f"got unrecognized dtype: {meta.dtype}")


def get_random_state(pyom2_lib=None, extra_settings=None):
    """Generates random Veros and PyOM states (for testing)"""
    from veros.core import numerics, streamfunction

    if extra_settings is None:
        extra_settings = {}

    state = get_default_state()
    settings = state.settings

    with settings.unlock():
        settings.update(extra_settings)

    state.initialize_variables()
    state.variables.__locked__ = False  # leave variables unlocked

    for var, meta in state.var_meta.items():
        if not meta.active:
            continue

        if var in ("tau", "taup1", "taum1"):
            continue

        val = _generate_random_var(state, var)
        setattr(state.variables, var, val)

    # ensure that masks and geometries are consistent with grid spacings
    numerics.calc_grid(state)
    numerics.calc_topo(state)

    streamfunction.streamfunction_init(state)

    if pyom2_lib is None:
        return state

    pyom_obj = load_pyom(pyom2_lib)
    pyom_obj = pyom_from_state(state, pyom_obj)

    return state, pyom_obj
