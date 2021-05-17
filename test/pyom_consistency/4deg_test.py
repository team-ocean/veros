from functools import partial
import numpy as np

from veros import tools
from veros.routines import veros_routine
from veros.pyom_compat import load_pyom, pyom_from_state, run_pyom
from veros.setups.global_4deg import GlobalFourDegreeSetup

from test_base import compare_state


class GlobalFourDegreeTest(GlobalFourDegreeSetup):
    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        super().set_parameter(state)
        settings.runlen = settings.dt_tracer * 5
        settings.restart_output_filename = None

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings
        super().set_initial_conditions(state)

        vs.surface_taux = vs.surface_taux / settings.rho_0
        vs.forc_tke_surface = vs.forc_tke_surface * settings.rho_0 ** 1.5

    @veros_routine
    def set_diagnostics(self, state):
        state.diagnostics.clear()


def set_forcing_pyom(pyom_obj, t_rest, t_star):
    m = pyom_obj.main_module

    year_in_seconds = 360 * 86400.
    time = m.itt * m.dt_tracer
    (n1, f1), (n2, f2) = tools.get_periodic_interval(time, year_in_seconds,
                                                        year_in_seconds / 12., 12)

    # wind stress
    m.surface_taux[...] = (f1 * self.taux[:, :, n1] + f2 * self.taux[:, :, n2])
    m.surface_tauy[...] = (f1 * self.tauy[:, :, n1] + f2 * self.tauy[:, :, n2])

    # tke flux
    t = pyom_obj.tke_module
    if t.enable_tke:
        t.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (m.surface_taux[1:-1, 1:-1] +
                                                            m.surface_taux[:-2, 1:-1]))**2 +
                                                    (0.5 * (m.surface_tauy[1:-1, 1:-1] +
                                                            m.surface_tauy[1:-1, :-2]))**2)**(3. / 2.)
    # heat flux : W/m^2 K kg/J m^3/kg = K m/s
    cp_0 = 3991.86795711963
    sst = f1 * self.sst_clim[:, :, n1] + f2 * self.sst_clim[:, :, n2]
    qnec = f1 * self.qnec[:, :, n1] + f2 * self.qnec[:, :, n2]
    qnet = f1 * self.qnet[:, :, n1] + f2 * self.qnet[:, :, n2]
    m.forc_temp_surface[...] = (qnet + qnec * (sst - m.temp[:, :, -1, m.tau - 1])) \
        * m.maskT[:, :, -1] / cp_0 / m.rho_0

    # salinity restoring
    t_rest = 30 * 86400.0
    sss = f1 * self.sss_clim[:, :, n1] + f2 * self.sss_clim[:, :, n2]
    m.forc_salt_surface[:] = 1. / t_rest * \
        (sss - m.salt[:, :, -1, m.tau - 1]) * m.maskT[:, :, -1] * m.dzt[-1]

    # apply simple ice mask
    mask = np.logical_and(m.temp[:, :, -1, m.tau - 1] * m.maskT[:, :, -1] < -1.8,
                            m.forc_temp_surface < 0.)
    m.forc_temp_surface[mask] = 0.0
    m.forc_salt_surface[mask] = 0.0

    if m.enable_tempsalt_sources:
        m.temp_source[:] = m.maskT * self.rest_tscl * \
            (f1 * t_star[:, :, :, n1] + f2 * t_star[:, :, :, n2] -
                m.temp[:, :, :, m.tau - 1])
        m.salt_source[:] = self.maskT * self.rest_tscl * \
            (f1 * self.s_star[:, :, :, n1] + f2 * self.s_star[:, :, :, n2] -
                m.salt[:, :, :, m.tau - 1])


def test_4deg(pyom2_lib):
    sim = GlobalFourDegreeTest()
    sim.setup()

    pyom_obj = load_pyom(pyom2_lib)
    pyom_obj = pyom_from_state(sim.state, pyom_obj, ignore_attrs=("t_star", "t_rest"))

    t_rest = sim.state.variables.t_rest
    t_star = sim.state.variables.t_star

    sim.run()
    run_pyom(pyom_obj, partial(set_forcing_pyom, t_rest=t_rest, t_star=t_star))

    compare_state(sim.state, pyom_obj)
