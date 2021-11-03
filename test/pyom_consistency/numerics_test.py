import pytest

from veros.core import numerics
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
            coord_degree=False,
            eq_of_state_type=1,
        ),
    )


def test_calc_grid(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(numerics.calc_grid(vs_state))
    pyom_obj.calc_grid()
    compare_state(vs_state, pyom_obj)


def test_calc_topo(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(numerics.calc_topo(vs_state))
    pyom_obj.calc_topo()
    compare_state(vs_state, pyom_obj)


def test_calc_beta(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(numerics.calc_beta(vs_state))
    pyom_obj.calc_beta()
    compare_state(vs_state, pyom_obj)


def test_calc_initial_conditions(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(numerics.calc_initial_conditions(vs_state))
    pyom_obj.calc_initial_conditions()
    compare_state(vs_state, pyom_obj)
