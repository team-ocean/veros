import pytest

from veros.state import VerosSettings, VerosVariables, VerosState


@pytest.fixture
def dummy_state():
    from veros.variables import VARIABLES, DIM_TO_SHAPE_VAR
    from veros.settings import SETTINGS

    return VerosState(VARIABLES, SETTINGS, DIM_TO_SHAPE_VAR)


@pytest.fixture
def dummy_settings():
    from veros.settings import SETTINGS

    return VerosSettings(SETTINGS)


@pytest.fixture
def dummy_variables():
    from veros.variables import VARIABLES, DIM_TO_SHAPE_VAR
    from veros.settings import SETTINGS

    dummy_state = VerosState(VARIABLES, SETTINGS, DIM_TO_SHAPE_VAR)
    dummy_state.initialize_variables()
    return dummy_state.variables


def test_lock_settings(dummy_settings):
    orig_val = dummy_settings.dt_tracer

    with pytest.raises(RuntimeError):
        dummy_settings.dt_tracer = 0

    assert dummy_settings.dt_tracer == orig_val

    with dummy_settings.unlock():
        dummy_settings.dt_tracer = 1

    assert dummy_settings.dt_tracer == 1


def test_settings_repr(dummy_settings):
    with dummy_settings.unlock():
        dummy_settings.dt_tracer = 1

    assert "dt_tracer = 1.0," in repr(dummy_settings)


def test_variables_repr(dummy_variables):
    from veros.core.operators import numpy as npx

    array_type = type(npx.array([]))
    assert f"tau = {array_type} with shape (), dtype int32," in repr(dummy_variables)


def test_to_xarray(dummy_state):
    pytest.importorskip("xarray")

    dummy_state.initialize_variables()
    ds = dummy_state.to_xarray()

    # settings
    assert tuple(ds.attrs.keys()) == tuple(dummy_state.settings.fields())
    assert tuple(ds.attrs.values()) == tuple(dummy_state.settings.values())

    # dimensions
    used_dims = set()
    for var, meta in dummy_state.var_meta.items():
        if var in dummy_state.variables:
            if meta.dims is None:
                continue

            used_dims |= set(meta.dims)

    assert set(ds.coords.keys()) == used_dims

    for dim in used_dims:
        assert int(ds.dims[dim]) == dummy_state.dimensions[dim]

    # variables
    for var in dummy_state.variables.fields():
        assert var in ds


def test_variable_init(dummy_state):
    with pytest.raises(RuntimeError):
        dummy_state.variables

    dummy_state.initialize_variables()

    assert isinstance(dummy_state.variables, VerosVariables)

    with pytest.raises(RuntimeError):
        dummy_state.initialize_variables()


def test_set_dimension(dummy_state):
    with dummy_state.settings.unlock():
        dummy_state.settings.nx = 10

    assert dummy_state.dimensions["xt"] == 10

    dummy_state.dimensions["foobar"] = 42
    assert dummy_state.dimensions["foobar"] == 42

    with pytest.raises(RuntimeError):
        dummy_state.dimensions["xt"] = 11

    assert dummy_state._dimensions["xt"] == "nx"


def test_resize_dimension(dummy_state):
    from veros.state import resize_dimension

    with dummy_state.settings.unlock():
        dummy_state.settings.nx = 10

    dummy_state.initialize_variables()

    assert dummy_state.dimensions["xt"] == 10
    assert dummy_state.variables.dxt.shape == (14,)

    resize_dimension(dummy_state, "xt", 100)

    assert dummy_state.dimensions["xt"] == 100
    assert dummy_state.variables.dxt.shape == (104,)


def test_timers(dummy_state):
    from veros.timer import Timer

    timer = dummy_state.timers["foobar"]
    assert isinstance(timer, Timer)
