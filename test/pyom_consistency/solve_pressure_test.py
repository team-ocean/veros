

from matplotlib import pyplot as plt
import numpy as np

import pytest

from veros.core import streamfunction, utilities
from veros.pyom_compat import get_random_state
from veros.core.streamfunction.solve_pressure import prepare_forcing
from veros.core.streamfunction.pressure_solvers import get_linear_solver
from test_base import compare_state


@pytest.fixture(autouse=True)
def ensure_diskless():
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "pyom_compatibility_mode", True)

@pytest.fixture
def random_state(pyom2_lib):
    return get_random_state(
        pyom2_lib,
        extra_settings=dict(
            nx=60,
            ny=60,
            nz=50,
            dt_tracer=3600,
            dt_mom=3600,
            enable_cyclic_x= True,
            enable_free_surface = True,
            enable_streamfunction = False,
        ),
    )


def test_solve_pressure(random_state):
    vs_state, pyom_obj = random_state
    m = pyom_obj.main_module
    vs = vs_state.variables
    settings = vs_state.settings

    compare_state(vs_state,pyom_obj)
    vs_state.variables.update(streamfunction.solve_pressure.solve_pressure(vs_state))
    #Initial guess in pyOM should be cyclical
    m.psi[:,:,vs.tau] = utilities.enforce_boundaries(m.psi[:,:,vs.tau],settings.enable_cyclic_x)
    m.psi[:,:,vs.taum1] = utilities.enforce_boundaries(m.psi[:,:,vs.taum1],settings.enable_cyclic_x)
    pyom_obj.solve_pressure()

    compare_state(vs_state,pyom_obj,rtol=2e-6)


