import pytest

from veros.core import momentum
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
        ),
    )


def test_momentum_advection(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(momentum.momentum_advection(vs_state))
    pyom_obj.momentum_advection()

    # not a part of momentum_advection in PyOM
    m = pyom_obj.main_module
    m.du[..., m.tau - 1] += m.du_adv
    m.dv[..., m.tau - 1] += m.dv_adv

    compare_state(vs_state, pyom_obj)


def test_vertical_velocity(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(momentum.vertical_velocity(vs_state))
    pyom_obj.vertical_velocity()
    compare_state(vs_state, pyom_obj)


def test_momentum(random_state):
    vs_state, pyom_obj = random_state

    # results are only identical if initial guess is already cyclic
    from veros.core import utilities

    vs = vs_state.variables
    m = pyom_obj.main_module
    m.psi[...] = utilities.enforce_boundaries(m.psi, vs_state.settings.enable_cyclic_x)
    vs.psi = utilities.enforce_boundaries(vs.psi, vs_state.settings.enable_cyclic_x)

    vs_state.variables.update(momentum.momentum(vs_state))
    pyom_obj.momentum()
    compare_state(vs_state, pyom_obj)
