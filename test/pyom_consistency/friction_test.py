from veros.core import friction
from veros.pyom_compat import get_random_state

from test_base import compare_state


TEST_SETTINGS = dict(
    nx=70,
    ny=60,
    nz=50,
    dt_tracer=3600,
    dt_mom=3600,
    enable_cyclic_x=True,
    enable_conserve_energy=True,
    enable_bottom_friction_var=True,
    enable_hor_friction_cos_scaling=True,
    enable_momentum_sources=True,
    r_ray=1,
    r_bot=1,
    r_quad_bot=1,
    A_h=1,
    A_hbi=1,
)


def test_explicit_vert_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.explicit_vert_friction(vs_state))
    pyom_obj.explicit_vert_friction()
    compare_state(vs_state, pyom_obj)


def test_implicit_vert_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.implicit_vert_friction(vs_state))
    pyom_obj.implicit_vert_friction()
    compare_state(vs_state, pyom_obj)


def test_rayleigh_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.rayleigh_friction(vs_state))
    pyom_obj.rayleigh_friction()
    compare_state(vs_state, pyom_obj)


def test_linear_bottom_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.linear_bottom_friction(vs_state))
    pyom_obj.linear_bottom_friction()
    compare_state(vs_state, pyom_obj)


def test_quadratic_bottom_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.quadratic_bottom_friction(vs_state))
    pyom_obj.quadratic_bottom_friction()
    compare_state(vs_state, pyom_obj)


def test_harmonic_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.harmonic_friction(vs_state))
    pyom_obj.harmonic_friction()
    compare_state(vs_state, pyom_obj)


def test_biharmonic_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.biharmonic_friction(vs_state))
    pyom_obj.biharmonic_friction()
    compare_state(vs_state, pyom_obj)


def test_momentum_sources(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(friction.momentum_sources(vs_state))
    pyom_obj.momentum_sources()
    compare_state(vs_state, pyom_obj)
