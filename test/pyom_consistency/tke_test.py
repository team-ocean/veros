import pytest

from veros.core import tke
from veros.pyom_compat import get_random_state

from test_base import compare_state

TEST_SETTINGS = dict(
    nx=70,
    ny=60,
    nz=50,
    dt_tracer=3600,
    dt_mom=3600,
    enable_cyclic_x=True,
    enable_idemix=True,
    tke_mxl_choice=2,
    enable_tke=True,
    enable_eke=True,
    enable_store_cabbeling_heat=True,
    enable_store_bottom_friction_tke=False,
    enable_tke_hor_diffusion=True,
    enable_tke_superbee_advection=True,
    enable_tke_upwind_advection=True,
)

PROBLEM_SETS = {
    "tke": dict(enable_tke=True),
    "no_tke": dict(enable_tke=False),
}


@pytest.mark.parametrize("problem_set", PROBLEM_SETS)
def test_set_tke_diffusivities(pyom2_lib, problem_set):
    settings = {**TEST_SETTINGS, **PROBLEM_SETS[problem_set]}
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=settings)

    vs_state.variables.update(tke.set_tke_diffusivities(vs_state))
    pyom_obj.set_tke_diffusivities()
    compare_state(vs_state, pyom_obj)


def test_integrate_tke(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)

    vs_state.variables.update(tke.integrate_tke(vs_state))
    pyom_obj.integrate_tke()
    compare_state(vs_state, pyom_obj)
