import pytest

from veros.state import VerosSettings


@pytest.fixture
def dummy_settings():
    from veros.settings import SETTINGS

    return VerosSettings(SETTINGS)


def test_lock_settings(dummy_settings):
    orig_val = dummy_settings.dt_tracer

    with pytest.raises(RuntimeError):
        dummy_settings.dt_tracer = 0

    assert dummy_settings.dt_tracer == orig_val

    with dummy_settings.unlock():
        dummy_settings.dt_tracer = 1

    assert dummy_settings.dt_tracer == 1
