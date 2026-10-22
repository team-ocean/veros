from types import SimpleNamespace

import numpy as np
import pytest

from veros.diagnostics import api


class CountingDiagnostic:
    def __init__(self, sampling_frequency, output_frequency):
        self.sampling_frequency = sampling_frequency
        self.output_frequency = output_frequency
        self.diagnose_steps = []
        self.output_steps = []
        self.step = 0

    def diagnose(self, state):
        self.diagnose_steps.append(self.step)

    def output(self, state):
        self.output_steps.append(self.step)


@pytest.mark.parametrize("steps_per_period", [7, 48])
def test_diagnostics_trigger_once_per_period(steps_per_period):
    # dt is not exactly representable for 7 steps per day, so the accumulated time drifts by roundoff
    frequency = 86400.0
    dt = frequency / steps_per_period
    diagnostic = CountingDiagnostic(sampling_frequency=frequency, output_frequency=frequency)
    state = SimpleNamespace(
        variables=SimpleNamespace(time=0.0),
        settings=SimpleNamespace(dt_tracer=dt),
        diagnostics={"counter": diagnostic},
    )

    num_periods = 2000
    for step in range(1, num_periods * steps_per_period + 1):
        state.variables.time = state.variables.time + dt
        diagnostic.step = step
        api.diagnose(state)
        api.output(state)

    expected = steps_per_period * np.arange(1, num_periods + 1)
    np.testing.assert_array_equal(diagnostic.diagnose_steps, expected)
    np.testing.assert_array_equal(diagnostic.output_steps, expected)
