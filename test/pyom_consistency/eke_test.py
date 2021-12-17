from veros.core import eke
from veros.pyom_compat import get_random_state

from test_base import compare_state


TEST_SETTINGS = dict(
    nx=70,
    ny=60,
    nz=50,
    dt_tracer=3600,
    dt_mom=3600,
    enable_cyclic_x=True,
    enable_eke=True,
    enable_TEM_friction=True,
    enable_eke_isopycnal_diffusion=True,
    enable_store_cabbeling_heat=True,
    enable_eke_superbee_advection=True,
    enable_eke_upwind_advection=True,
)


def test_set_eke_diffusivities(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(eke.set_eke_diffusivities(vs_state))
    pyom_obj.set_eke_diffusivities()
    compare_state(vs_state, pyom_obj)


def test_integrate_eke(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(eke.integrate_eke(vs_state))
    pyom_obj.integrate_eke()
    compare_state(vs_state, pyom_obj)
