import pytest

from veros.core import idemix
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
        ),
    )


def test_set_idemix_parameter(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(idemix.set_idemix_parameter(vs_state))
    pyom_obj.set_idemix_parameter()
    allowed_failures = ["c0"]  # computation of c0 uses several float literals in PyOM
    compare_state(vs_state, pyom_obj, allowed_failures=allowed_failures)


def test_integrate_idemix(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(idemix.integrate_idemix(vs_state))
    pyom_obj.integrate_idemix()
    compare_state(vs_state, pyom_obj)
