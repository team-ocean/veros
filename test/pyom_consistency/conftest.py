import pytest
import importlib


@pytest.fixture(autouse=True)
def ensure_pyom_compatibility():
    import veros
    importlib.reload(veros)
    veros.runtime_settings.pyom_compatibility_mode = True
