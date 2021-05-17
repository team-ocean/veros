import os
import numpy as np

from veros.setups.acc import ACCSetup


def _normalize(*arrays):
    if any(a.size == 0 for a in arrays):
        return arrays

    norm = np.abs(arrays[0]).max()
    if norm == 0.:
        return arrays

    return (a / norm for a in arrays)


def test_restart(tmpdir):
    os.chdir(tmpdir)

    timesteps_1 = 5
    timesteps_2 = 1

    dt_tracer = 86_400 / 2
    restart_file = "restart.h5"

    acc_no_restart = ACCSetup(override=dict(
        identifier="ACC_no_restart",
        restart_input_filename=None,
        restart_output_filename=restart_file,
        dt_tracer=dt_tracer,
        runlen=timesteps_1 * dt_tracer
    ))
    acc_no_restart.setup()
    acc_no_restart.run()

    acc_restart = ACCSetup(override=dict(
        identifier="ACC_restart",
        restart_input_filename=restart_file,
        restart_output_filename=None,
        dt_tracer=dt_tracer,
        runlen=timesteps_2 * dt_tracer
    ))
    acc_restart.setup()
    acc_restart.run()

    with acc_no_restart.state.settings.unlock():
        acc_no_restart.state.settings.runlen = timesteps_2 * dt_tracer

    acc_no_restart.run()

    state_1, state_2 = acc_restart.state, acc_no_restart.state

    for setting in state_1.settings.fields():
        if setting in ("identifier", "restart_input_filename", "restart_output_filename", "runlen"):
            continue

        s1 = state_1.settings.get(setting)
        s2 = state_2.settings.get(setting)
        assert s1 == s2

    def check_var(var):
        v1 = state_1.variables.get(var)
        v2 = state_2.variables.get(var)
        np.testing.assert_allclose(*_normalize(v1, v2), atol=1e-6, rtol=0)

    for var in state_1.variables.fields():
        if var in ("itt",):
            continue

        # salt is not used by this setup, contains only numerical noise
        if "salt" in var:
            continue

        check_var(var)
