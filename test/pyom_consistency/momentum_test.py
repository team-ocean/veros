import pytest

from veros.core import momentum

from test_base import get_random_state, compare_state


@pytest.fixture
def random_state(pyom2_lib):
    return get_random_state(pyom2_lib, extra_settings=dict(
        nx=70,
        ny=60,
        nz=50,
        coord_degree=True,
        enable_cyclic_x=True,
        enable_conserve_energy=True,
        enable_bottom_friction_var=True,
        enable_hor_friction_cos_scaling=True,
        enable_implicit_vert_friction=True,
        enable_explicit_vert_friction=True,
        enable_TEM_friction=True,
        enable_hor_friction=True,
        enable_biharmonic_friction=True,
        enable_ray_friction=True,
        enable_bottom_friction=True,
        enable_quadratic_bottom_friction=True,
        enable_momentum_sources=True,
        congr_epsilon=1e-12,
        congr_max_iterations=10000,
    ))


def test_momentum_advection(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(momentum.momentum_advection(vs_state))
    pyom_obj.momentum_advection()
    compare_state(vs_state, pyom_obj)


def test_vertical_velocity(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(momentum.vertical_velocity(vs_state))
    pyom_obj.vertical_velocity()
    compare_state(vs_state, pyom_obj)


def test_momentum(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(momentum.momentum(vs_state))
    pyom_obj.momentum()
    compare_state(vs_state, pyom_obj)
