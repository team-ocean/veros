import numpy as np
import pytest

from veros import runtime_settings
from veros.core.operators import numpy as npx
from veros.state import VerosState


@pytest.fixture
def diagnostic_state():
    from veros.settings import SETTINGS
    from veros.variables import VARIABLES, DIM_TO_SHAPE_VAR

    previous = runtime_settings.diskless_mode, runtime_settings.use_io_threads
    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "use_io_threads", False)
    state = VerosState(VARIABLES, SETTINGS, dict(DIM_TO_SHAPE_VAR, isle=1))
    with state.settings.unlock():
        state.settings.nx = 4
        state.settings.ny = 4
        state.settings.nz = 3
    state.initialize_variables()
    with state.variables.unlock():
        for key in ("maskT", "maskU", "maskV", "maskW", "area_t", "dzt", "dxt", "dyt", "dxu", "dyu", "cost", "cosu"):
            setattr(state.variables, key, npx.ones_like(state.variables.get(key)))
        state.variables.temp = npx.ones_like(state.variables.temp) * 3
        state.variables.salt = npx.ones_like(state.variables.salt) * 5
    try:
        yield state
    finally:
        object.__setattr__(runtime_settings, "diskless_mode", previous[0])
        object.__setattr__(runtime_settings, "use_io_threads", previous[1])


def test_average_finalize_resets_window(diagnostic_state):
    from veros.diagnostics.averages import Averages

    state = diagnostic_state
    diagnostic = Averages(state)
    diagnostic.output_variables = ("temp", "time")
    diagnostic.initialize(state)
    with state.variables.unlock():
        state.variables.time = 10
    diagnostic.diagnose(state)
    with state.variables.unlock():
        state.variables.temp = state.variables.temp * 3
        state.variables.time = 20
    diagnostic.diagnose(state)
    values = diagnostic.finalize(state)
    np.testing.assert_array_equal(values["temp"], 6)
    np.testing.assert_array_equal(values["time"], 15)
    assert int(diagnostic.variables.average_nitts) == 0
    np.testing.assert_array_equal(diagnostic.variables.temp, 0)
    empty = diagnostic.finalize(state)
    for value in empty.values():
        np.testing.assert_array_equal(value, 0)
    np.testing.assert_array_equal(values["temp"], 6)


@pytest.mark.parametrize("kind", ("energy", "overturning"))
def test_finalize_preserves_static_values_and_resets_means(diagnostic_state, kind):
    from veros.diagnostics.energy import Energy
    from veros.diagnostics.overturning import Overturning

    state = diagnostic_state
    diagnostic = Energy(state) if kind == "energy" else Overturning(state)
    diagnostic.extra_dimensions = {"sigma": 4}
    diagnostic.initialize_variables(state)
    for key in diagnostic.output_variables:
        setattr(diagnostic.variables, key, npx.ones_like(diagnostic.variables.get(key)) * 12)
    diagnostic.variables.nitts = 3
    values = diagnostic.finalize(state)
    means = diagnostic.output_variables if kind == "energy" else ("trans", "vsf_iso", "vsf_depth")
    for key in diagnostic.output_variables:
        expected = 4 * state.settings.rho_0 if kind == "energy" else 4 if key in means else 12
        np.testing.assert_array_equal(values[key], expected)
        np.testing.assert_array_equal(diagnostic.variables.get(key), 0 if key in means else 12)
    assert int(diagnostic.variables.nitts) == 0
    empty = diagnostic.finalize(state)
    for key in means:
        np.testing.assert_array_equal(empty[key], 0)


def test_tracer_finalize_updates_history_without_logging(diagnostic_state, monkeypatch):
    from veros import logger
    from veros.diagnostics.tracer_monitor import TracerMonitor

    state = diagnostic_state
    diagnostic = TracerMonitor(state)
    diagnostic.initialize(state)
    messages = []
    monkeypatch.setattr(logger, "diagnostic", messages.append)
    first = diagnostic.finalize(state)
    assert messages == []
    np.testing.assert_allclose([first[key] for key in ("temp", "salt", "vtemp", "vsalt")], [3, 5, 9, 25])
    second = diagnostic.finalize(state)
    np.testing.assert_allclose([second[key] for key in ("dtemp", "dsalt", "dvtemp", "dvsalt")], 0)
    diagnostic.write_finalized(state, first)
    assert len(messages) == 4
    np.testing.assert_array_equal(diagnostic.variables.tempm1, 3 * 4 * 4 * 3)


def test_cfl_finalize_defers_nan_error_to_writer(diagnostic_state, monkeypatch):
    from veros import logger
    from veros.diagnostics.cfl_monitor import CFLMonitor

    state = diagnostic_state
    diagnostic = CFLMonitor(state)
    messages = []
    monkeypatch.setattr(logger, "diagnostic", messages.append)
    values = diagnostic.finalize(state)
    np.testing.assert_array_equal([values["cfl"], values["wcfl"]], 0)
    assert messages == []
    diagnostic.write_finalized(state, values)
    assert len(messages) == 2
    with state.variables.unlock():
        state.variables.u = state.variables.u + npx.nan
    values = diagnostic.finalize(state)
    assert np.isnan(values["cfl"])
    with pytest.raises(RuntimeError, match="CFL number is NaN"):
        diagnostic.write_finalized(state, values)


def test_snapshot_finalized_output_matches_eager_output(diagnostic_state, tmp_path):
    h5netcdf = pytest.importorskip("h5netcdf")
    from veros.diagnostics.snapshot import Snapshot

    state = diagnostic_state
    diagnostic = Snapshot(state)
    diagnostic.output_variables = ("temp", "time", "dxt")
    diagnostic.output_frequency = 1
    diagnostic.initialize(state)
    with state.variables.unlock():
        state.variables.time = 86400
        state.variables.temp = npx.arange(state.variables.temp.size).reshape(state.variables.temp.shape)
    values = diagnostic.finalize(state)
    object.__setattr__(runtime_settings, "diskless_mode", False)
    diagnostic.output_path = str(tmp_path / "eager.nc")
    diagnostic.initialize_output(state)
    diagnostic.write_output(state)
    with state.variables.unlock():
        state.variables.time = 172800
        state.variables.tau = (state.variables.tau + 1) % 3
        state.variables.temp = state.variables.temp * 0
    diagnostic.output_path = str(tmp_path / "finalized.nc")
    diagnostic.write_finalized(state, values, time_value=86400)
    with h5netcdf.File(tmp_path / "eager.nc", "r") as eager, h5netcdf.File(tmp_path / "finalized.nc", "r") as finalized:
        for key in ("Time", *diagnostic.output_variables):
            np.testing.assert_array_equal(finalized.variables[key][:], eager.variables[key][:])
    np.testing.assert_array_equal(state.variables.temp, 0)
    assert int(state.variables.time) == 172800
