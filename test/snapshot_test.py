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
    snapshot_class = Snapshot

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
        self.step_clocks = []
        snapshot = state.diagnostics["snapshot"] = self.snapshot_class(state)
        snapshot.output_variables = ["temp", "salt", "dtemp", "dsalt"]
        snapshot.output_frequency = state.settings.dt_tracer
        averages = state.diagnostics["averages"]
        averages.output_variables = ["temp", "salt"]
        averages.sampling_frequency = state.settings.dt_tracer
        averages.output_frequency = 2 * state.settings.dt_tracer
        state.diagnostics["tracer_monitor"].output_frequency = state.settings.dt_tracer

    @veros_routine
    def after_timestep(self, state):
        vs = state.variables
        self.step_clocks.append((int(vs.itt), float(vs.time)))
        # The clock still describes tau; the completed step is available at taup1.
        for slot, seconds in ((vs.tau, vs.time), (vs.taup1, vs.time + state.settings.dt_tracer)):
            np.testing.assert_allclose(vs.temp[2:-2, 2:-2, -1, slot], 10.0 + seconds * 0.01)
            np.testing.assert_allclose(vs.salt[2:-2, 2:-2, -1, slot], 35.0 + seconds * 0.001)


class CustomSnapshot(Snapshot):
    def output(self, state):
        super().output(state)


@pytest.mark.parametrize(
    "dt, runlen, output_frequency, automatic_times",
    (
        (10.0, 60.0, 10.0, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        (10.0, 60.0, 20.0, [0.0, 20.0, 40.0]),
        (10.0, 60.0, 15.0, [0.0, 20.0, 30.0, 50.0]),
        (10.0, 60.0, 120.0, [0.0]),
        (10.0, 10.0, 10.0, [0.0]),
        (0.125, 0.75, 0.25, [0.0, 0.25, 0.5]),
    ),
    ids=("every-step", "every-other-step", "nonaligned-interval", "long-interval", "one-step", "fractional-step"),
)
@pytest.mark.parametrize("snapshot_class", (Snapshot, CustomSnapshot))
def test_snapshot_matches_model_time(
    tmp_path, monkeypatch, dt, runlen, output_frequency, automatic_times, snapshot_class
):
    monkeypatch.chdir(tmp_path)
    simulation = ForcedBoxSetup(override=dict(dt_tracer=dt, dt_mom=dt, runlen=runlen))
    simulation.snapshot_class = snapshot_class
    simulation.setup()
    snapshot = simulation.state.diagnostics["snapshot"]
    snapshot.output_frequency = output_frequency
    snapshot.output(simulation.state)

    simulation.run(show_progress_bar=False)
    snapshot.output(simulation.state)

    with h5netcdf.File("forced_box.snapshot.nc") as outfile:
        expected_times = [0.0, *automatic_times, runlen]
        np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, expected_times)
        # A 10 m surface cell gains flux * time / thickness, without advection or mixing.
        for record, seconds in enumerate(expected_times):
            np.testing.assert_allclose(np.asarray(outfile["temp"])[record, -1], 10.0 + seconds * 0.01)
            np.testing.assert_allclose(np.asarray(outfile["salt"])[record, -1], 35.0 + seconds * 0.001)
        np.testing.assert_allclose(np.asarray(outfile["temp"])[:, :-1], 10.0)
        np.testing.assert_allclose(np.asarray(outfile["salt"])[:, :-1], 35.0)

    vs = simulation.state.variables
    assert vs.time == runlen
    assert vs.itt == round(runlen / dt)
    assert simulation.step_clocks == [(step, step * dt) for step in range(round(runlen / dt))]
    np.testing.assert_allclose(vs.temp[2:-2, 2:-2, -1, vs.tau], 10.0 + runlen * 0.01)
    np.testing.assert_allclose(vs.salt[2:-2, 2:-2, -1, vs.tau], 35.0 + runlen * 0.001)


def test_averages_sample_input_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = ForcedBoxSetup()
    simulation.setup()
    simulation.run(show_progress_bar=False)

    with h5netcdf.File("forced_box.averages.nc") as outfile:
        np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, [0.0, 20.0, 40.0])
        for record, (temp, salt) in enumerate(((10.0, 35.0), (10.15, 35.015), (10.35, 35.035))):
            np.testing.assert_allclose(np.asarray(outfile["temp"])[record, -1], temp)
            np.testing.assert_allclose(np.asarray(outfile["salt"])[record, -1], salt)
        np.testing.assert_allclose(np.asarray(outfile["temp"])[:, :-1], 10.0)
        np.testing.assert_allclose(np.asarray(outfile["salt"])[:, :-1], 35.0)

    # The sample at 50 seconds is retained for a later output or restart.
    averages_vs = simulation.state.diagnostics["averages"].variables
    assert averages_vs.average_nitts == 1
    np.testing.assert_allclose(averages_vs.temp[2:-2, 2:-2, -1], 10.5)
    np.testing.assert_allclose(averages_vs.salt[2:-2, 2:-2, -1], 35.05)


def test_tracer_monitor_outputs_before_rotation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = ForcedBoxSetup()
    simulation.setup()
    simulation.run(show_progress_bar=False)

    tracer_vs = simulation.state.diagnostics["tracer_monitor"].variables
    volume = 4 * 4 * 3 * 1000.0**2 * 10.0
    # The last automatic output is at 50 seconds, before the final rotation.
    np.testing.assert_allclose(tracer_vs.tempm1 / volume, (10.0 + 10.0 + 10.5) / 3)
    np.testing.assert_allclose(tracer_vs.saltm1 / volume, (35.0 + 35.0 + 35.05) / 3)


def test_averages_sampling_uses_starting_clock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = ForcedBoxSetup()
    simulation.setup()
    averages = simulation.state.diagnostics["averages"]
    averages.sampling_frequency = 20.0
    averages.output_frequency = 40.0
    simulation.run(show_progress_bar=False)

    with h5netcdf.File("forced_box.averages.nc") as outfile:
        np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, [0.0, 40.0])
        np.testing.assert_allclose(np.asarray(outfile["temp"])[0, -1], 10.0)
        np.testing.assert_allclose(np.asarray(outfile["temp"])[1, -1], 10.3)
    assert averages.variables.average_nitts == 0


def test_zero_length_run_has_no_automatic_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = ForcedBoxSetup(override=dict(runlen=0.0))
    simulation.setup()
    simulation.run(show_progress_bar=False)

    with h5netcdf.File("forced_box.snapshot.nc") as outfile:
        assert outfile.dimensions["Time"].size == 0
    assert simulation.state.variables.time == 0.0
    assert simulation.state.variables.itt == 0


def test_time_dependent_forcing_uses_starting_clock(tmp_path, monkeypatch):
    class TimeDependentSetup(ForcedBoxSetup):
        @veros_routine
        def set_forcing(self, state):
            super().set_forcing(state)
            vs = state.variables
            vs.forc_temp_surface = update(vs.forc_temp_surface, at[...], 0.1 * (1 + vs.time / 10.0))

        @veros_routine
        def after_timestep(self, state):
            pass

    monkeypatch.chdir(tmp_path)
    simulation = TimeDependentSetup()
    simulation.setup()
    simulation.run(show_progress_bar=False)
    simulation.state.diagnostics["snapshot"].output(simulation.state)

    with h5netcdf.File("forced_box.snapshot.nc") as outfile:
        np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, [0, 10, 20, 30, 40, 50, 60])
        for record, temp in enumerate((10.0, 10.1, 10.3, 10.6, 11.0, 11.5, 12.1)):
            np.testing.assert_allclose(np.asarray(outfile["temp"])[record, -1], temp)


def test_snapshot_timing_after_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = ForcedBoxSetup(override=dict(identifier="first", runlen=20.0, restart_output_filename="restart.h5"))
    first.setup()
    first.state.diagnostics["snapshot"].output_frequency = 20.0
    first.run(show_progress_bar=False)

    resumed = ForcedBoxSetup(override=dict(identifier="resumed", runlen=40.0, restart_input_filename="restart.h5"))
    resumed.setup()
    assert resumed.state.variables.time == 20.0
    resumed.state.diagnostics["snapshot"].output_frequency = 20.0
    resumed.run(show_progress_bar=False)
    assert resumed.step_clocks == [(0, 20.0), (1, 30.0), (2, 40.0), (3, 50.0)]

    for identifier, times, temperatures in (("first", [0.0], [10.0]), ("resumed", [20.0, 40.0], [10.2, 10.4])):
        with h5netcdf.File(f"{identifier}.snapshot.nc") as outfile:
            np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, times)
            for record, temp in enumerate(temperatures):
                np.testing.assert_allclose(np.asarray(outfile["temp"])[record, -1], temp)
    with h5netcdf.File("resumed.averages.nc") as outfile:
        np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, [20.0, 40.0])
        np.testing.assert_allclose(np.asarray(outfile["temp"])[0, -1], 10.15)
        np.testing.assert_allclose(np.asarray(outfile["temp"])[1, -1], 10.35)
    assert resumed.state.variables.time == 60.0
    np.testing.assert_allclose(resumed.state.variables.temp[2:-2, 2:-2, -1, resumed.state.variables.tau], 10.6)


def test_failed_output_preserves_input_clock_and_indices(tmp_path, monkeypatch):
    class FailingSnapshot(Snapshot):
        def output(self, state):
            raise OSError("snapshot output failed")

    monkeypatch.chdir(tmp_path)
    simulation = ForcedBoxSetup()
    simulation.snapshot_class = FailingSnapshot
    simulation.setup()
    vs = simulation.state.variables
    slots = (vs.taum1, vs.tau, vs.taup1)

    with pytest.raises(OSError, match="snapshot output failed"):
        simulation.step(simulation.state)

    assert vs.time == 0.0
    assert vs.itt == 0
    assert (vs.taum1, vs.tau, vs.taup1) == slots
    np.testing.assert_allclose(vs.temp[2:-2, 2:-2, -1, vs.tau], 10.0)
    np.testing.assert_allclose(vs.temp[2:-2, 2:-2, -1, vs.taup1], 10.1)


@pytest.mark.parametrize("slots", ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
def test_snapshot_uses_tau_for_all_variables(tmp_path, monkeypatch, slots):
    monkeypatch.chdir(tmp_path)
    state = get_default_state()
    with state.settings.unlock():
        state.settings.update(nx=4, ny=4, nz=3, enable_tke=True, enable_eke=True, enable_idemix=True, dt_tracer=10.0)
    state.var_meta["custom_state"] = Variable("Custom state", T_GRID + TIMESTEPS)
    state.var_meta["custom_rate"] = Variable("Custom tendency", T_GRID + TIMESTEPS)
    state.initialize_variables()
    resize_dimension(state, "isle", 2)

    timestep_variables = (
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
        "dtemp",
        "dsalt",
        "du",
        "dv",
        "dHd",
        "dtke",
        "deke",
        "dE_iw",
        "dpsin",
        "custom_rate",
    )
    with state.variables.unlock():
        vs = state.variables
        vs.taum1, vs.tau, vs.taup1 = slots
        vs.time = 10.0
        for name in ("maskT", "maskU", "maskV", "maskW", "maskZ"):
            setattr(vs, name, npx.ones_like(getattr(vs, name)))
        for name in timestep_variables:
            setattr(vs, name, npx.broadcast_to(npx.array([11.0, 22.0, 33.0]), getattr(vs, name).shape))
        vs.forc_temp_surface = npx.full_like(vs.forc_temp_surface, 7.0)

    snapshot = Snapshot(state)
    snapshot.output_variables = [*timestep_variables, "forc_temp_surface"]
    snapshot.output_frequency = 10.0
    averages = Averages(state)
    averages.output_variables = ["temp"]
    averages.sampling_frequency = 10.0
    averages.output_frequency = 10.0
    state.diagnostics.update(snapshot=snapshot, averages=averages)
    diagnostics.initialize(state)

    snapshot.output(state)
    diagnostics.diagnose(state)
    diagnostics.output(state)
    with state.variables.unlock():
        vs.taum1, vs.tau, vs.taup1 = vs.tau, vs.taup1, vs.taum1
        vs.time = vs.time + state.settings.dt_tracer
    snapshot.output(state)

    old_value = (11.0, 22.0, 33.0)[slots[1]]
    completed_value = (11.0, 22.0, 33.0)[slots[2]]
    with h5netcdf.File(snapshot.get_output_file_name(state)) as outfile:
        for name in timestep_variables:
            for record, expected in enumerate((old_value, old_value, completed_value)):
                np.testing.assert_allclose(np.asarray(outfile[name])[record], expected, err_msg=name)
        np.testing.assert_allclose(np.asarray(outfile["forc_temp_surface"]), 7.0)
        np.testing.assert_allclose(np.asarray(outfile["Time"]) * 86400.0, [10.0, 10.0, 20.0])

    with h5netcdf.File(averages.get_output_file_name(state)) as outfile:
        np.testing.assert_allclose(np.asarray(outfile["temp"])[0], old_value)
    assert (vs.taum1, vs.tau, vs.taup1) == (slots[1], slots[2], slots[0])
