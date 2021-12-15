import numpy as np
from textwrap import indent

from veros.variables import remove_ghosts
from veros.pyom_compat import state_from_pyom, VEROS_TO_PYOM_SETTING, VEROS_TO_PYOM_VAR


def _normalize(*arrays):
    if any(a.size == 0 for a in arrays):
        return arrays

    norm = np.nanmax(np.abs(arrays[0]))

    if norm == 0.0:
        return arrays

    return tuple(a / norm for a in arrays)


def compare_state(
    vs_state,
    pyom_obj,
    atol=1e-10,
    rtol=1e-8,
    include_ghosts=False,
    allowed_failures=None,
    normalize=False,
):
    IGNORE_SETTINGS = ("congr_max_iterations",)

    if allowed_failures is None:
        allowed_failures = []

    pyom_state = state_from_pyom(pyom_obj)

    def assert_setting(setting):
        vs_val = vs_state.settings.get(setting)
        setting = VEROS_TO_PYOM_SETTING.get(setting, setting)
        if setting is None or setting in IGNORE_SETTINGS:
            return

        pyom_val = pyom_state.settings.get(setting)
        assert vs_val == pyom_val, (vs_val, pyom_val)

    def assert_var(var):
        vs_val = vs_state.variables.get(var)

        var = VEROS_TO_PYOM_VAR.get(var, var)
        if var is None:
            return

        if var not in pyom_state.variables:
            return

        pyom_val = pyom_state.variables.get(var)

        if var in ("tau", "taup1", "taum1"):
            assert pyom_val == vs_val + 1
            return

        if not include_ghosts:
            vs_val = remove_ghosts(vs_val, vs_state.var_meta[var].dims)
            pyom_val = remove_ghosts(pyom_val, pyom_state.var_meta[var].dims)

        if normalize:
            vs_val, pyom_val = _normalize(vs_val, pyom_val)

        np.testing.assert_allclose(vs_val, pyom_val, atol=atol, rtol=rtol)

    passed = True

    for setting in vs_state.settings.fields():
        try:
            assert_setting(setting)
        except AssertionError as exc:
            if setting not in allowed_failures:
                print(f"{setting}:{indent(str(exc), ' ' * 4)}")
                passed = False

    for var in vs_state.variables.fields():
        try:
            assert_var(var)
        except AssertionError as exc:
            if var not in allowed_failures:
                print(f"{var}:{indent(str(exc), ' ' * 4)}")
                passed = False

    assert passed
