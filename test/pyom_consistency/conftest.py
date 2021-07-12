import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker("forked")


@pytest.fixture(autouse=True)
def setup_logging():
    from veros.logs import setup_logging

    setup_logging(loglevel="warning")


@pytest.fixture(autouse=True)
def ensure_pyom_compatibility():
    import veros

    object.__setattr__(veros.runtime_settings, "pyom_compatibility_mode", True)
    try:
        yield
    finally:
        object.__setattr__(veros.runtime_settings, "pyom_compatibility_mode", False)
