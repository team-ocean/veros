import pytest
import numpy as np

from veros.core import advection

from test_base import get_random_state, compare_state


@pytest.fixture
def random_state(pyom2_lib):
    return get_random_state(pyom2_lib, extra_settings=dict(
        nx=70,
        ny=60,
        nz=50,
        dt_tracer=3600,
    ))


def test_calculate_velocity_on_wgrid(random_state):
    vs_state, pyom_obj = random_state
    advection.calculate_velocity_on_wgrid(vs_state)
    pyom_obj.calculate_velocity_on_wgrid()

    compare_state(vs_state, pyom_obj)


def test_adv_flux_2nd(random_state):
    vs_state, pyom_obj = random_state

    res = advection.adv_flux_2nd(vs_state, vs_state.variables.Hd[..., 1])

    m = pyom_obj.main_module
    pyom_obj.adv_flux_2nd(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz,
        adv_fe=m.flux_east,
        adv_fn=m.flux_north,
        adv_ft=m.flux_top,
        var=m.hd[..., 1]
    )

    np.testing.assert_allclose(res[0], m.flux_east)
    np.testing.assert_allclose(res[1], m.flux_north)
    np.testing.assert_allclose(res[2], m.flux_top)


def test_adv_flux_superbee(random_state):
    vs_state, pyom_obj = random_state

    res = advection.adv_flux_superbee(vs_state, vs_state.variables.Hd[..., 1])

    m = pyom_obj.main_module
    pyom_obj.adv_flux_superbee(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz,
        adv_fe=m.flux_east,
        adv_fn=m.flux_north,
        adv_ft=m.flux_top,
        var=m.hd[..., 1]
    )

    np.testing.assert_allclose(res[0], m.flux_east)
    np.testing.assert_allclose(res[1], m.flux_north)
    np.testing.assert_allclose(res[2], m.flux_top)


def test_adv_flux_upwind_wgrid(random_state):
    vs_state, pyom_obj = random_state

    res = advection.adv_flux_upwind_wgrid(vs_state, vs_state.variables.Hd[..., 1])

    m = pyom_obj.main_module
    pyom_obj.adv_flux_upwind_wgrid(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz,
        adv_fe=m.flux_east,
        adv_fn=m.flux_north,
        adv_ft=m.flux_top,
        var=m.hd[..., 1]
    )

    np.testing.assert_allclose(res[0], m.flux_east)
    np.testing.assert_allclose(res[1], m.flux_north)
    np.testing.assert_allclose(res[2], m.flux_top)


def test_adv_flux_superbee_wgrid(random_state):
    vs_state, pyom_obj = random_state

    res = advection.adv_flux_superbee_wgrid(vs_state, vs_state.variables.Hd[..., 1])

    m = pyom_obj.main_module
    pyom_obj.adv_flux_superbee_wgrid(
        is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz,
        adv_fe=m.flux_east,
        adv_fn=m.flux_north,
        adv_ft=m.flux_top,
        var=m.hd[..., 1]
    )

    np.testing.assert_allclose(res[0], m.flux_east)
    np.testing.assert_allclose(res[1], m.flux_north)
    np.testing.assert_allclose(res[2], m.flux_top)
