from veros.core import thermodynamics
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
    enable_hor_friction_cos_scaling=True,
    enable_tempsalt_sources=True,
    enable_hor_diffusion=True,
    enable_superbee_advection=True,
    enable_tke=True,
    enable_biharmonic_mixing=True,
    enable_neutral_diffusion=True,
    enable_skew_diffusion=True,
    enable_TEM_friction=True,
    eq_of_state_type=1,
)


def prepare_inputs(vs_state, pyom_obj):
    # implementations are only identical if non-water values are 0
    vs = vs_state.variables
    for var in (
        "P_diss_sources",
        "P_diss_hmix",
    ):
        getattr(pyom_obj.main_module, var.lower())[...] *= vs.maskT
        with vs.unlock():
            setattr(vs, var, vs.get(var) * vs.maskT)

    return vs_state, pyom_obj


def test_thermodynamics(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state, pyom_obj = prepare_inputs(vs_state, pyom_obj)

    vs_state.variables.update(thermodynamics.thermodynamics(vs_state))
    pyom_obj.thermodynamics()

    compare_state(vs_state, pyom_obj)
