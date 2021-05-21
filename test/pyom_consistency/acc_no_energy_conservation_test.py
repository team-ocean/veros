from veros.routines import veros_routine
from veros.pyom_compat import load_pyom, pyom_from_state, run_pyom
from veros.setups.acc import ACCSetup

from test_base import compare_state


class ACCTestNoEnergyConservation(ACCSetup):
    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        super().set_parameter(state)

        settings.runlen = settings.dt_tracer * 100
        settings.restart_output_filename = None
        settings.enable_conserve_energy = False

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


def test_acc(pyom2_lib):
    sim = ACCTestNoEnergyConservation()
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

    compare_state(
        sim.state,
        pyom_obj,
        normalize=True,
        rtol=0,
        atol=1e-4,
        allowed_failures=("salt", "dsalt", "dsalt_vmix", "dsalt_iso"),
    )
