import pytest

from veros.routines import veros_routine
from veros.pyom_compat import load_pyom, pyom_from_state, run_pyom
from veros.setups.acc import ACCSetup

from test_base import compare_state


TEST_SETS = {
    "standard": dict(),
    "pressure": dict(enable_streamfunction=False),
    "no-energy-conservation": dict(enable_conserve_energy=False),
}


class ACCTest(ACCSetup):
    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        super().set_parameter(state)

        settings.runlen = settings.dt_tracer * 100
        settings.restart_output_filename = None

        # do not exist in pyOM
        settings.kappaH_min = 0.0
        settings.enable_kappaH_profile = False
        settings.enable_Prandtl_tke = True

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings
        super().set_initial_conditions(state)

        vs.surface_taux = vs.surface_taux / settings.rho_0

    @veros_routine
    def set_diagnostics(self, state):
        state.diagnostics.clear()


@pytest.mark.parametrize("test_set", TEST_SETS.keys())
def test_acc(pyom2_lib, test_set):
    extra_settings = TEST_SETS[test_set]

    sim = ACCTest(override=extra_settings)
    sim.setup()

    pyom_obj = load_pyom(pyom2_lib)
    pyom_obj = pyom_from_state(sim.state, pyom_obj, ignore_attrs=("t_star", "t_rest"))

    t_rest = sim.state.variables.t_rest
    t_star = sim.state.variables.t_star

    sim.run()

    def set_forcing_pyom(pyom_obj):
        m = pyom_obj.main_module
        m.forc_temp_surface[:] = t_rest * (t_star - m.temp[:, :, -1, m.tau - 1])

    run_pyom(pyom_obj, set_forcing_pyom)

    # salt is not used by this setup
    allowed_failures = ("salt", "dsalt", "dsalt_vmix", "dsalt_iso")
    atol = 1e-8

    if test_set == "pressure":
        # pressure setups are more numerically sensitive, stick to "observables"
        atol = 1e-5
        allowed_failures = set(sim.state.variables.fields()) - {"u", "v", "temp", "psi"}

    compare_state(
        sim.state,
        pyom_obj,
        atol=atol,
        rtol=0,
        normalize=True,
        allowed_failures=allowed_failures,
    )
