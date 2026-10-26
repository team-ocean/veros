import h5netcdf
import numpy as np
import pytest

from veros import VerosSetup, diagnostics, veros_routine
from veros.core.operators import at, numpy as npx, update
from veros.diagnostics.averages import Averages
from veros.diagnostics.snapshot import Snapshot
from veros.state import get_default_state, resize_dimension
from veros.variables import TIMESTEPS, T_GRID, Variable


class ForcedBoxSetup(VerosSetup):
    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        settings.identifier = "forced_box"
        settings.nx, settings.ny, settings.nz = 4, 4, 3
        settings.dt_tracer = settings.dt_mom = 10.0
        settings.runlen = 60.0
        settings.coord_degree = False
        settings.enable_streamfunction = False
        settings.restart_output_filename = None

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        vs.dxt = update(vs.dxt, at[...], 1000.0)
        vs.dyt = update(vs.dyt, at[...], 1000.0)
        vs.dzt = update(vs.dzt, at[...], 10.0)

    @veros_routine
    def set_coriolis(self, state):
        pass

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        vs.kbot = update(vs.kbot, at[...], 1)

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        vs.temp = update(vs.temp, at[...], 10.0)
        vs.salt = update(vs.salt, at[...], 35.0)

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.forc_temp_surface = update(vs.forc_temp_surface, at[...], 0.1)
        vs.forc_salt_surface = update(vs.forc_salt_surface, at[...], 0.01)

    @veros_routine
    def set_diagnostics(self, state):
        snapshot = state.diagnostics["snapshot"]
        snapshot.output_variables = ["temp", "salt", "dtemp", "dsalt"]
        snapshot.output_frequency = state.settings.dt_tracer

    @veros_routine
    def after_timestep(self, state):
        pass


@pytest.mark.parametrize("output_frequency", (10.0, 20.0))
def test_snapshot_matches_model_time(tmp_path, monkeypatch, output_frequency):
    monkeypatch.chdir(tmp_path)
    simulation = ForcedBoxSetup()
    simulation.setup()
    snapshot = simulation.state.diagnostics["snapshot"]
    snapshot.output_frequency = output_frequency
    snapshot.output(simulation.state)

    simulation.run(show_progress_bar=False)
    snapshot.output(simulation.state)

    with h5netcdf.File("forced_box.snapshot.nc") as outfile:
        expected_times = [0.0, *np.arange(output_frequency, 61.0, output_frequency), 60.0]
        np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, expected_times)
        # A 10 m surface cell gains flux * time / thickness, without advection or mixing.
        for record, seconds in enumerate(expected_times):
            np.testing.assert_allclose(np.asarray(outfile["temp"])[record, -1], 10.0 + seconds * 0.01)
            np.testing.assert_allclose(np.asarray(outfile["salt"])[record, -1], 35.0 + seconds * 0.001)
        np.testing.assert_allclose(np.asarray(outfile["temp"])[:, :-1], 10.0)
        np.testing.assert_allclose(np.asarray(outfile["salt"])[:, :-1], 35.0)

    vs = simulation.state.variables
    np.testing.assert_allclose(vs.temp[2:-2, 2:-2, -1, vs.tau], 10.6)
    np.testing.assert_allclose(vs.salt[2:-2, 2:-2, -1, vs.tau], 35.06)


@pytest.mark.parametrize("slots", ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
def test_snapshot_slot_selection(tmp_path, monkeypatch, slots):
    monkeypatch.chdir(tmp_path)
    state = get_default_state()
    with state.settings.unlock():
        state.settings.update(nx=4, ny=4, nz=3, enable_tke=True, enable_eke=True, enable_idemix=True, dt_tracer=10.0)
    state.var_meta["custom_state"] = Variable("Custom state", T_GRID + TIMESTEPS)
    state.var_meta["custom_rate"] = Variable("Custom tendency", T_GRID + TIMESTEPS)
    state.initialize_variables()
    resize_dimension(state, "isle", 2)

    completed_variables = (
        "temp",
        "salt",
        "u",
        "v",
        "w",
        "rho",
        "Nsqr",
        "Hd",
        "int_drhodT",
        "int_drhodS",
        "psi",
        "dpsi",
        "tke",
        "eke",
        "E_iw",
        "custom_state",
    )
    tendency_variables = ("dtemp", "dsalt", "du", "dv", "dHd", "dtke", "deke", "dE_iw", "dpsin", "custom_rate")
    with state.variables.unlock():
        vs = state.variables
        vs.taum1, vs.tau, vs.taup1 = slots
        vs.time = 10.0
        for name in ("maskT", "maskU", "maskV", "maskW", "maskZ"):
            setattr(vs, name, npx.ones_like(getattr(vs, name)))
        for name in completed_variables + tendency_variables:
            setattr(vs, name, npx.broadcast_to(npx.array([11.0, 22.0, 33.0]), getattr(vs, name).shape))
        vs.forc_temp_surface = npx.full_like(vs.forc_temp_surface, 7.0)

    snapshot = Snapshot(state)
    snapshot.output_variables = [*completed_variables, *tendency_variables, "forc_temp_surface"]
    snapshot.tendency_variables.add("custom_rate")
    snapshot.output_frequency = 10.0
    averages = Averages(state)
    averages.output_variables = ["temp"]
    averages.output_frequency = 10.0
    state.diagnostics.update(snapshot=snapshot, averages=averages)
    diagnostics.initialize(state)

    snapshot.output(state)
    averages.diagnose(state)
    diagnostics.output(state, time_index=vs.taup1)
    diagnostics.output(state)

    old_value = (11.0, 22.0, 33.0)[slots[1]]
    completed_value = (11.0, 22.0, 33.0)[slots[2]]
    with h5netcdf.File(snapshot.get_output_file_name(state)) as outfile:
        for name in completed_variables:
            for record, expected in enumerate((old_value, completed_value, old_value)):
                np.testing.assert_allclose(np.asarray(outfile[name])[record], expected, err_msg=name)
        for name in tendency_variables:
            np.testing.assert_allclose(np.asarray(outfile[name]), old_value, err_msg=name)
        np.testing.assert_allclose(np.asarray(outfile["forc_temp_surface"]), 7.0)

    with h5netcdf.File(averages.get_output_file_name(state)) as outfile:
        np.testing.assert_allclose(np.asarray(outfile["temp"])[0], old_value)
    assert (vs.taum1, vs.tau, vs.taup1) == slots
