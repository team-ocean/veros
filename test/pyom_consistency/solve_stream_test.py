import pytest

from veros.core import streamfunction
from veros.pyom_compat import get_random_state

from test_base import compare_state


@pytest.fixture
def random_state(pyom2_lib):
    return get_random_state(
        pyom2_lib,
        extra_settings=dict(
            nx=70,
            ny=60,
            nz=50,
            dt_tracer=3600,
            dt_mom=3600,
            enable_cyclic_x=True,
            congr_epsilon=1e-12,
            congr_max_iterations=10_000,
        ),
    )


def test_solve_streamfunction(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(streamfunction.solve_streamfunction(vs_state))
    pyom_obj.solve_streamfunction()
    compare_state(vs_state, pyom_obj)
