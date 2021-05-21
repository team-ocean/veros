import click

from veros import veros_routine
from veros.pyom_compat import load_pyom, pyom_from_state, run_pyom
from veros.setups.acc import ACCSetup


class ACC2Benchmark(ACCSetup):
    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        super().set_parameter(state)

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


@click.option("--pyom2-lib", type=click.Path(readable=True, dir_okay=False), default=None)
@click.option("--timesteps", type=int, default=100)
def main(pyom2_lib, timesteps):
    sim = ACC2Benchmark()
    sim.setup()

    settings = sim.state.settings

    with settings.unlock():
        settings.runlen = timesteps * settings.dt_tracer

    if not pyom2_lib:
        sim.run()
        return

    pyom_obj = load_pyom(pyom2_lib)
    pyom_obj = pyom_from_state(sim.state, pyom_obj, ignore_attrs=("t_star", "t_rest"))

    t_rest = sim.state.variables.t_rest
    t_star = sim.state.variables.t_star

    def set_forcing_pyom(pyom_obj):
        m = pyom_obj.main_module
        m.forc_temp_surface[:] = t_rest * (t_star - m.temp[:, :, -1, m.tau - 1])

    run_pyom(pyom_obj, set_forcing_pyom)


if __name__ == "__main__":
    main()
