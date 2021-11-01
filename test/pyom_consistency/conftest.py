import sys
import importlib

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker("forked")


@pytest.fixture(autouse=True)
def setup_test():
    import veros
    from veros.logs import update_logging

    update_logging(loglevel="warning")
    object.__setattr__(veros.runtime_settings, "pyom_compatibility_mode", True)

    # reload all core modules to make sure changes take effect
    for name, mod in list(sys.modules.items()):
        if name.startswith("veros.core"):
            importlib.reload(mod)

    try:
        yield
    finally:
        object.__setattr__(veros.runtime_settings, "pyom_compatibility_mode", False)

    for name, mod in list(sys.modules.items()):
        if name.startswith("veros.core"):
            importlib.reload(mod)
