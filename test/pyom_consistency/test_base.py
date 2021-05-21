import numpy as np
from textwrap import indent

from veros.state import get_default_state
from veros.variables import remove_ghosts, get_shape
from veros.pyom_compat import load_pyom, state_from_pyom, pyom_from_state, VEROS_TO_PYOM_SETTING, VEROS_TO_PYOM_VAR


def _normalize(*arrays):
    if any(a.size == 0 for a in arrays):
        return arrays

    norm = np.nanmax(np.abs(arrays[0]))

    if norm == 0.0:
        return arrays

    return tuple(a / norm for a in arrays)


def compare_state(
    vs_state, pyom_obj, atol=1e-8, rtol=1e-6, include_ghosts=False, allowed_failures=None, normalize=False
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

        if np.issubdtype(type(vs_val), np.floating):
            # pyOM's settings only have float precision
            vs_val = np.float32(vs_val)
            pyom_val = np.float32(pyom_val)

        assert vs_val == pyom_val

    for setting in vs_state.settings.fields():
        assert_setting(setting)

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
    for var in vs_state.variables.fields():
        try:
            assert_var(var)
        except AssertionError as exc:
            if var not in allowed_failures:
                print(f"{var}:{indent(str(exc), ' ' * 4)}")
                passed = False

    assert passed


def _generate_random_var(state, var):
    meta = state.var_meta[var]
    shape = get_shape(state.dimensions, meta.dims)

    if var == "kbot":
        val = np.zeros(shape)
        val[2:-2, 2:-2] = np.random.randint(1, state.dimensions["zt"], size=(shape[0] - 4, shape[1] - 4))
        island_mask = np.random.choice(val[2:-2, 2:-2].size, size=10)
        val[2:-2, 2:-2].flat[island_mask] = 0
        return val

    if var in ("dxt", "dxu", "dyt", "dyu"):
        if state.settings.coord_degree:
            val = 1 + 1e-2 * np.random.randn(*shape)
        else:
            val = 10e3 + 100 * np.random.randn(*shape)
        return val

    if var in ("dzt", "dzw"):
        val = 100 + np.random.randn(*shape)
        return val

    if np.issubdtype(np.dtype(meta.dtype), np.floating):
        val = np.random.randn(*shape)
        if var in ("salt",):
            val = np.abs(val)

        return val

    if np.issubdtype(np.dtype(meta.dtype), np.integer):
        val = np.random.randint(0, 100, size=shape)
        return val

    if np.issubdtype(np.dtype(meta.dtype), np.bool_):
        return np.random.randint(0, 1, size=shape, dtype="bool")

    raise TypeError(f"got unrecognized dtype: {meta.dtype}")


def get_random_state(pyom2_lib, extra_settings=None):
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

    pyom_obj = load_pyom(pyom2_lib)
    pyom_obj = pyom_from_state(state, pyom_obj)

    return state, pyom_obj
