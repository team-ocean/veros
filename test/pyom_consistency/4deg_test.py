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

        settings.runlen = settings.dt_tracer * 100
        settings.restart_output_filename = None

        # do not exist in pyOM
        settings.kappaH_min = 0.0
        settings.enable_kappaH_profile = False
        settings.enable_Prandtl_tke = True

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        settings = state.settings
        super().set_forcing(state)

        vs.surface_taux = vs.surface_taux / settings.rho_0
        vs.surface_tauy = vs.surface_tauy / settings.rho_0

    @veros_routine
    def set_diagnostics(self, state):
        state.diagnostics.clear()


def set_forcing_pyom(pyom_obj, vs_state):
    vs = vs_state.variables
    m = pyom_obj.main_module

    year_in_seconds = 360 * 86400.0
    time = m.itt * m.dt_tracer
    (n1, f1), (n2, f2) = tools.get_periodic_interval(time, year_in_seconds, year_in_seconds / 12.0, 12)

    # wind stress
    m.surface_taux[...] = (f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]) / m.rho_0
    m.surface_tauy[...] = (f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]) / m.rho_0

    # tke flux
    t = pyom_obj.tke_module
    if t.enable_tke:
        t.forc_tke_surface[1:-1, 1:-1] = np.sqrt(
            (0.5 * (m.surface_taux[1:-1, 1:-1] + m.surface_taux[:-2, 1:-1])) ** 2
            + (0.5 * (m.surface_tauy[1:-1, 1:-1] + m.surface_tauy[1:-1, :-2])) ** 2
        ) ** (3.0 / 2.0)
    # heat flux : W/m^2 K kg/J m^3/kg = K m/s
    cp_0 = 3991.86795711963
    sst = f1 * vs.sst_clim[:, :, n1] + f2 * vs.sst_clim[:, :, n2]
    qnec = f1 * vs.qnec[:, :, n1] + f2 * vs.qnec[:, :, n2]
    qnet = f1 * vs.qnet[:, :, n1] + f2 * vs.qnet[:, :, n2]
    m.forc_temp_surface[...] = (qnet + qnec * (sst - m.temp[:, :, -1, m.tau - 1])) * m.maskt[:, :, -1] / cp_0 / m.rho_0

    # salinity restoring
    t_rest = 30 * 86400.0
    sss = f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
    m.forc_salt_surface[:] = 1.0 / t_rest * (sss - m.salt[:, :, -1, m.tau - 1]) * m.maskt[:, :, -1] * m.dzt[-1]

    # apply simple ice mask
    mask = np.logical_and(m.temp[:, :, -1, m.tau - 1] * m.maskt[:, :, -1] < -1.8, m.forc_temp_surface < 0.0)
    m.forc_temp_surface[mask] = 0.0
    m.forc_salt_surface[mask] = 0.0

    if m.enable_tempsalt_sources:
        m.temp_source[:] = (
            m.maskt
            * vs.rest_tscl
            * (f1 * vs.t_star[:, :, :, n1] + f2 * vs.t_star[:, :, :, n2] - m.temp[:, :, :, m.tau - 1])
        )
        m.salt_source[:] = (
            m.maskt
            * vs.rest_tscl
            * (f1 * vs.s_star[:, :, :, n1] + f2 * vs.s_star[:, :, :, n2] - m.salt[:, :, :, m.tau - 1])
        )


def test_4deg(pyom2_lib):
    sim = GlobalFourDegreeTest()
    sim.setup()

    pyom_obj = load_pyom(pyom2_lib)
    pyom_obj = pyom_from_state(
        sim.state, pyom_obj, ignore_attrs=("taux", "tauy", "sss_clim", "sst_clim", "qnec", "qnet")
    )

    sim.run()
    run_pyom(pyom_obj, partial(set_forcing_pyom, vs_state=sim.state))

    # test passes if differences are less than 0.1% of the maximum value of each variable
    compare_state(
        sim.state,
        pyom_obj,
        normalize=True,
        rtol=0,
        atol=1e-4,
        allowed_failures=("Ai_ez", "Ai_nz", "Ai_bx", "Ai_by"),
    )
