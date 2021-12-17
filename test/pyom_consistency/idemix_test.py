from veros.core import idemix
from veros.pyom_compat import get_random_state

from test_base import compare_state


TEST_SETTINGS = dict(
    nx=70,
    ny=60,
    nz=50,
    dt_tracer=3600,
    dt_mom=3600,
    enable_idemix=True,
    enable_idemix_hor_diffusion=True,
    enable_idemix_superbee_advection=True,
    enable_idemix_upwind_advection=True,
    enable_eke=True,
    enable_store_cabbeling_heat=True,
    enable_eke_diss_bottom=True,
    enable_eke_diss_surfbot=True,
    enable_store_bottom_friction_tke=True,
    enable_TEM_friction=True,
)


def test_set_idemix_parameter(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(idemix.set_idemix_parameter(vs_state))
    pyom_obj.set_idemix_parameter()
    compare_state(vs_state, pyom_obj)


def test_integrate_idemix(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(idemix.integrate_idemix(vs_state))
    pyom_obj.integrate_idemix()
    compare_state(vs_state, pyom_obj)
