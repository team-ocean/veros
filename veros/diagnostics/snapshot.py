import os
import copy

from veros import time, logger
from veros.diagnostics.base import VerosDiagnostic

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
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    tendency_variables = frozenset({"dtemp", "dsalt", "du", "dv", "dHd", "dtke", "deke", "dE_iw", "dpsin"})
    """Raw tendencies stored at ``tau`` before rotation. Add custom variables with this
    convention to the instance's set. ``dpsi`` is also a raw tendency, but the current
    streamfunction solve stores it at ``taup1``.
    """

    def __init__(self, state):
        self.output_variables = []
        self.tendency_variables = set(self.tendency_variables)

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

    def output(self, state, *, time_index=None):
        """Write the selected model slot, defaulting to ``tau`` for manual calls.

        Automatic output before rotation supplies ``taup1`` for completed states
        and the latest ``dpsi``. Raw tendencies listed in ``tendency_variables``
        retain ``tau``; they are neither states at Time nor realized interval rates.
        """
        vs = state.variables

        time_length, time_unit = time.format_time(vs.time)
        logger.info(f" Writing snapshot at {time_length:.2f} {time_unit}")

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        time_indices = {key: time_index for key in self.output_variables if key not in self.tendency_variables}
        self.write_output(state, time_indices=time_indices)
