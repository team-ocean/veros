import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--pyom2-lib", default=None,
        help="Path to PyOM2 library (must be given for consistency tests)"
    )
    parser.addoption(
        "--backend", choices=["numpy", "jax"], default="numpy",
        help="Numerical backend to test"
    )


def pytest_configure(config):
    backend = config.getoption("--backend")
    os.environ["VEROS_BACKEND"] = backend


def pytest_collection_modifyitems(config, items):
    if config.getoption("--pyom2-lib"):
        return

    skip = pytest.mark.skip(reason="need --pyom2-lib option to run")
    for item in items:
        if "pyom2_lib" in item.fixturenames:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.pyom2_lib
    if "pyom2_lib" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("pyom2_lib", [option_value])


@pytest.fixture(autouse=True)
def set_random_seed():
    import numpy as np
    np.random.seed(17)
