import pytest

from veros.core import isoneutral

from test_base import get_random_state, compare_state


@pytest.fixture
def random_state(pyom2_lib):
    return get_random_state(pyom2_lib, extra_settings=dict(
        nx=70,
        ny=60,
        nz=50,
        enable_neutral_diffusion=True,
        enable_skew_diffusion=True,
        enable_TEM_friction=True,
        K_iso_steep=1,

    ))


def test_isoneutral_diffusion_pre(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(isoneutral.isoneutral_diffusion_pre(vs_state))
    pyom_obj.isoneutral_diffusion_pre()
    compare_state(vs_state, pyom_obj)


def test_isoneutral_diag_streamfunction(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(isoneutral.isoneutral_diag_streamfunction(vs_state))
    pyom_obj.isoneutral_diag_streamfunction()
    compare_state(vs_state, pyom_obj)


@pytest.mark.parametrize('istemp', [True, False])
def test_isoneutral_diffusion(random_state, istemp):
    vs_state, pyom_obj = random_state
    m = pyom_obj.main_module
    vs_state.variables.update(isoneutral.isoneutral_diffusion(vs_state, vs_state.variables.temp, istemp))
    pyom_obj.isoneutral_diffusion(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz,
        tr=m.temp, istemp=istemp
    )
    compare_state(vs_state, pyom_obj)


@pytest.mark.parametrize('istemp', [True, False])
def test_isoneutral_skew_diffusion(random_state, istemp):
    vs_state, pyom_obj = random_state
    m = pyom_obj.main_module
    vs_state.variables.update(isoneutral.isoneutral_skew_diffusion(vs_state, vs_state.variables.temp, istemp))
    pyom_obj.isoneutral_skew_diffusion(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz,
        tr=m.temp, istemp=istemp
    )
    compare_state(vs_state, pyom_obj)


def test_isoneutral_friction(random_state):
    vs_state, pyom_obj = random_state
    vs_state.variables.update(isoneutral.isoneutral_friction(vs_state))
    pyom_obj.isoneutral_friction()
    compare_state(vs_state, pyom_obj)
