import numpy as np
import pytest

from veros import runtime_settings, veros_routine


def test_nonintegral_diagnostic_schedule_and_empty_windows():
    if runtime_settings.backend != "jax":
        pytest.skip("compiled diagnostic scheduling requires JAX")
    from veros.setups.acc_basic import ACCBasicSetup

    class ScheduleSetup(ACCBasicSetup):
        @veros_routine
        def set_diagnostics(self, state):
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = None
                diagnostic.sampling_frequency = None
            averages = state.diagnostics["averages"]
            averages.output_variables = ("temp",)
            averages.sampling_frequency = 1200.0
            averages.output_frequency = 900.0

        @veros_routine(supports_compilation=True)
        def _step_dynamics(self, state):
            state.variables.time += state.settings.dt_tracer
            state.variables.itt += 1

    previous = runtime_settings.diskless_mode
    object.__setattr__(runtime_settings, "diskless_mode", True)
    try:
        sim = ScheduleSetup(override=dict(nx=8, ny=8, dt_mom=600, dt_tracer=600))
        sim.setup()
        with sim.state.variables.unlock():
            sim.state.variables.time = 150.0
            sim.state.variables.temp = sim.state.variables.temp * 0 + 3
        prepared = sim.prepare_integrator(record_diagnostics=("averages", "snapshot"))
        initial = prepared.pack(sim.state)
        final, records = prepared.advance(initial, n_steps=7)
        np.testing.assert_array_equal(records["averages"].valid, [False, True, True, False, True, True, False])
        np.testing.assert_array_equal(records["snapshot"].valid, False)
        np.testing.assert_array_equal(np.asarray(records["averages"].values["temp"])[[1, 4, 5]], 3)
        np.testing.assert_array_equal(np.asarray(records["averages"].values["temp"])[2], 0)
        first, _ = prepared.advance(initial, n_steps=4)
        split, _ = prepared.advance(first, n_steps=3)
        np.testing.assert_array_equal(final.diagnostics["averages"].temp, split.diagnostics["averages"].temp)
        assert float(final.diagnostics["averages"].average_nitts) == 0
        # Sampling without output retains a window across a call boundary.
        sim.state.diagnostics["averages"].output_frequency = None
        sim.state.diagnostics["averages"].sampling_frequency = 600.0
        prepared = sim.prepare_integrator(record_diagnostics=("averages",))
        carry, records = prepared.advance(prepared.pack(sim.state), n_steps=2)
        assert float(carry.diagnostics["averages"].average_nitts) == 2
        np.testing.assert_array_equal(carry.diagnostics["averages"].temp, 6)
        np.testing.assert_array_equal(records["averages"].valid, False)
        sim.state.diagnostics["averages"].output_frequency = float("inf")
        prepared = sim.prepare_integrator(record_diagnostics=("averages",))
        carry, records = prepared.advance(prepared.pack(sim.state), n_steps=2)
        assert float(carry.diagnostics["averages"].average_nitts) == 2
        np.testing.assert_array_equal(records["averages"].valid, False)
    finally:
        object.__setattr__(runtime_settings, "diskless_mode", previous)
