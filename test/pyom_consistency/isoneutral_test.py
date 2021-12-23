import pytest

from veros.core import isoneutral
from veros.pyom_compat import get_random_state

from test_base import compare_state


TEST_SETTINGS = dict(
    nx=70,
    ny=60,
    nz=50,
    dt_tracer=3600,
    dt_mom=3600,
    enable_neutral_diffusion=True,
    enable_skew_diffusion=True,
    enable_TEM_friction=True,
    K_iso_steep=1,
)


def test_isoneutral_diffusion_pre(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(isoneutral.isoneutral_diffusion_pre(vs_state))
    pyom_obj.isoneutral_diffusion_pre()
    compare_state(vs_state, pyom_obj)


def test_isoneutral_diag_streamfunction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(isoneutral.isoneutral_diag_streamfunction(vs_state))
    pyom_obj.isoneutral_diag_streamfunction()
    compare_state(vs_state, pyom_obj)


@pytest.mark.parametrize("istemp", [True, False])
def test_isoneutral_diffusion(pyom2_lib, istemp):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    m = pyom_obj.main_module
    vs = vs_state.variables
    vs.update(isoneutral.isoneutral_diffusion(vs_state, vs.temp if istemp else vs.salt, istemp))
    pyom_obj.isoneutral_diffusion(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz, tr=m.temp if istemp else m.salt, istemp=istemp
    )
    compare_state(vs_state, pyom_obj)


@pytest.mark.parametrize("istemp", [True, False])
def test_isoneutral_skew_diffusion(pyom2_lib, istemp):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    m = pyom_obj.main_module
    vs = vs_state.variables
    vs.update(isoneutral.isoneutral_skew_diffusion(vs_state, vs.temp if istemp else vs.salt, istemp))
    pyom_obj.isoneutral_skew_diffusion(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz, tr=m.temp if istemp else m.salt, istemp=istemp
    )
    compare_state(vs_state, pyom_obj)


def test_isoneutral_friction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(isoneutral.isoneutral_friction(vs_state))
    pyom_obj.isoneutral_friction()
    compare_state(vs_state, pyom_obj)
