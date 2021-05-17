import pytest

from veros.core import thermodynamics

from test_base import get_random_state, compare_state


@pytest.fixture
def random_state(pyom2_lib):
    return get_random_state(pyom2_lib, extra_settings=dict(
        nx=70,
        ny=60,
        nz=50,
        enable_cyclic_x=True,
        enable_conserve_energy=True,
        enable_hor_friction_cos_scaling=True,
        enable_tempsalt_sources=True,
        enable_hor_diffusion=True,
        enable_superbee_advection=True,
        enable_tke=True,
        enable_biharmonic_mixing=True,
        enable_neutral_diffusion=True,
        enable_skew_diffusion=True,
        enable_TEM_friction=True,
    ))


def test_thermodynamics(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(thermodynamics.thermodynamics(vs_state))
    pyom_obj.thermodynamics()
    compare_state(vs_state, pyom_obj)
