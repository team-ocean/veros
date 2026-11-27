import copy

from veros import time, logger
from veros.diagnostics.base import VerosDiagnostic
from veros.variables import TIMESTEPS

DEFAULT_OUTPUT_VARS = [
    "dxt",
    "dxu",
    "dyt",
    "dyu",
    "zt",
    "zw",
    "dzt",
    "dzw",
    "ht",
    "hu",
    "hv",
    "beta",
    "area_t",
    "area_u",
    "area_v",
    "rho",
    "prho",
    "int_drhodT",
    "int_drhodS",
    "Nsqr",
    "Hd",
    "temp",
    "salt",
    "forc_temp_surface",
    "forc_salt_surface",
    "u",
    "v",
    "w",
    "p_hydro",
    "kappaM",
    "kappaH",
    "surface_taux",
    "surface_tauy",
    "forc_rho_surface",
    "psi",
    "isle",
    "psin",
    "xt",
    "xu",
    "yt",
    "yu",
    "temp_source",
    "salt_source",
    "u_source",
    "v_source",
    "tke",
    "forc_tke_surface",
    "eke",
    "E_iw",
    "forc_iw_surface",
    "forc_iw_bottom",
]


class Snapshot(VerosDiagnostic):
    """Writes snapshots of the current solution. Also reads and writes the main restart
    data required for restarting a Veros simulation.
    """

    output_path = "{identifier}.snapshot.nc"
    """File to write to. May contain format strings that are replaced with Veros attributes."""
    name = "snapshot"  #:
    supports_compilation = True
    output_frequency = None  #: Frequency (in seconds) in which output is written.

    def __init__(self, state):
        self.output_variables = []

        for var in DEFAULT_OUTPUT_VARS:
            active = state.var_meta[var].active
            if callable(active):
                active = active(state.settings)

            if active:
                self.output_variables.append(var)

    def initialize(self, state):
        vs = state.variables

        self.var_meta = {var: copy.copy(state.var_meta[var]) for var in self.output_variables}
        for var in self.var_meta.values():
            var.write_to_restart = False

        self.variables = vs
        self.initialize_output(state)

    def diagnose(self, state):
        pass

    def output(self, state):
        self.write_finalized(state, self.finalize(state))

    def write_finalized(self, state, values, time_value=None):
        time_length, time_unit = time.format_time(state.variables.time if time_value is None else time_value)
        logger.info(f" Writing snapshot at {time_length:.2f} {time_unit}")
        super().write_finalized(state, values, time_value=time_value)

    def finalize(self, state):
        vs = state.variables
        values = {}
        for key in self.output_variables:
            dims = self.var_meta[key].dims
            value = vs.get(key)
            if dims and any(dim in TIMESTEPS for dim in dims):
                value = value[tuple(vs.tau if dim in TIMESTEPS else slice(None) for dim in dims)]
            values[key] = value
        return values
