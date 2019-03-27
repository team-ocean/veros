import os
import tempfile
import numpy as np

from veros import VerosSetup, veros_method, settings, runtime_settings as rs

yt_start = -39.0
yt_end = 43
yu_start = -40.0
yu_end = 42


class ACC2(VerosSetup):
    """
    A simple global model with a Southern Ocean and Atlantic part
    """
    @veros_method
    def set_parameter(self, vs):
        vs.identifier = "acc2_restart_test"

        vs.nx, vs.ny, vs.nz = 30, 42, 15
        vs.dt_mom = 4800
        vs.dt_tracer = 86400 / 2.
        vs.runlen = 86400 * 365

        vs.coord_degree = True
        vs.enable_cyclic_x = True

        vs.congr_epsilon = 1e-12
        vs.congr_max_iterations = 5000

        vs.enable_neutral_diffusion = True
        vs.K_iso_0 = 1000.0
        vs.K_iso_steep = 500.0
        vs.iso_dslope = 0.005
        vs.iso_slopec = 0.01
        vs.enable_skew_diffusion = True

        vs.enable_hor_friction = True
        vs.A_h = (2 * vs.degtom) ** 3 * 2e-11
        vs.enable_hor_friction_cos_scaling = 1
        vs.hor_friction_cosPower = 1

        vs.enable_bottom_friction = True
        vs.r_bot = 1e-5

        vs.enable_implicit_vert_friction = True
        vs.enable_tke = True
        vs.c_k = 0.1
        vs.c_eps = 0.7
        vs.alpha_tke = 30.0
        vs.mxl_min = 1e-8
        vs.tke_mxl_choice = 2

        vs.K_gm_0 = 1000.0

        vs.enable_eke = True
        vs.eke_k_max = 1e4
        vs.eke_c_k = 0.4
        vs.eke_c_eps = 0.5
        vs.eke_cross = 2.
        vs.eke_crhin = 1.0
        vs.eke_lmin = 100.0
        vs.enable_eke_superbee_advection = True
        vs.enable_eke_isopycnal_diffusion = True

        vs.enable_idemix = True
        vs.enable_idemix_hor_diffusion = True
        vs.enable_eke_diss_surfbot = True
        vs.eke_diss_surfbot_frac = 0.2
        vs.enable_idemix_superbee_advection = True

        vs.eq_of_state_type = 3

    @veros_method
    def set_grid(self, vs):
        ddz = [50., 70., 100., 140., 190., 240., 290., 340.,
               390., 440., 490., 540., 590., 640., 690.]
        vs.dxt[:] = 2.0
        vs.dyt[:] = 2.0
        vs.x_origin = 0.0
        vs.y_origin = -40.0
        vs.dzt[:] = ddz[::-1]
        vs.dzt[:] *= 1 / 2.5

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[:, :] = 2 * vs.omega * np.sin(vs.yt[None, :] / 180. * vs.pi)

    @veros_method
    def set_topography(self, vs):
        (X, Y) = np.meshgrid(vs.xt, vs.yt)
        X = X.transpose()
        Y = Y.transpose()
        vs.kbot[...] = (X > 1.0) | (Y < -20)

    @veros_method
    def set_initial_conditions(self, vs):
        # initial conditions
        vs.temp[:, :, :, 0:2] = ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None]
        vs.salt[:, :, :, 0:2] = 35.0 * vs.maskT[..., None]

        # wind stress forcing
        taux = np.zeros(vs.ny + 1, dtype=vs.default_float_type)
        yt = vs.yt[2:vs.ny + 3]
        taux = (.1e-3 * np.sin(np.pi * (vs.yu[2:vs.ny + 3] - yu_start) / (-20.0 - yt_start))) * (yt < -20) \
            + (.1e-3 * (1 - np.cos(2 * np.pi * (vs.yu[2:vs.ny + 3] - 10.0) / (yu_end - 10.0)))) * (yt > 10)
        vs.surface_taux[:, 2:vs.ny + 3] = taux * vs.maskU[:, 2:vs.ny + 3, -1]

        # surface heatflux forcing
        vs.t_star = 15 * np.invert((vs.yt < -20) | (vs.yt > 20)) \
            + 15 * (vs.yt - yt_start) / (-20 - yt_start) * (vs.yt < -20) \
            + 15 * (1 - (vs.yt - 20) / (yt_end - 20)) * (vs.yt > 20.)
        vs.t_rest = vs.dzt[np.newaxis, -1] / (30. * 86400.) * vs.maskT[:, :, -1]

        if vs.enable_tke:
            vs.forc_tke_surface[2:-2, 2:-2] = np.sqrt((0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]))**2
                                                        + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]))**2)**(1.5)

        if vs.enable_idemix:
            vs.forc_iw_bottom[:] = 1.0e-6 * vs.maskW[:, :, -1]
            vs.forc_iw_surface[:] = 0.1e-6 * vs.maskW[:, :, -1]

    @veros_method
    def set_forcing(self, vs):
        vs.forc_temp_surface[:] = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    @veros_method
    def set_diagnostics(self, vs):
        pass

    def after_timestep(self, vs):
        pass


class RestartTest(object):
    timesteps = 10

    def __init__(self, backend):
        rs.backend = backend

        self.restart_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False).name

        self.acc_no_restart = ACC2()
        self.acc_no_restart.diskless_mode = True
        self.acc_restart = ACC2()
        self.acc_restart.state.restart_output_filename = self.restart_file

    def run(self):
        self.acc_no_restart.setup()
        self.acc_no_restart.state.runlen = self.acc_no_restart.state.dt_tracer * self.timesteps
        self.acc_no_restart.run()

        self.acc_restart.setup()
        self.acc_restart.state.runlen = (self.timesteps - 1) * self.acc_no_restart.state.dt_tracer
        self.acc_restart.run()

        self.acc_restart.restart_input_filename = self.restart_file
        self.acc_restart.setup()
        self.acc_restart.state.runlen = self.acc_no_restart.state.time - self.acc_restart.state.time
        self.acc_restart.run()

        os.remove(self.restart_file)
        return self.test_passed()

    def test_passed(self):
        passed = True
        for setting in settings.SETTINGS:
            s_1, s_2 = (getattr(obj, setting) for obj in (self.acc_no_restart.state, self.acc_restart.state))
            if s_1 != s_2:
                print(setting, s_1, s_2)
        for var in sorted(self.acc_no_restart.state.variables.keys()):
            if "salt" in var:
                continue
            arr_1, arr_2 = (getattr(obj, var) for obj in (self.acc_no_restart.state, self.acc_restart.state))
            try:
                arr_1 = arr_1.copy2numpy()
            except AttributeError:
                pass
            try:
                arr_2 = arr_2.copy2numpy()
            except AttributeError:
                pass
            # if "psi" in var:
            #     arr_1 = arr_1[3:-2, 2:-2]
            #     arr_2 = arr_2[3:-2, 2:-2]
            print(var)
            np.testing.assert_allclose(*self._normalize(arr_1, arr_2), atol=1e-7)
        return passed

    def _normalize(self, *arrays):
        if any(a.size == 0 for a in arrays):
            return arrays
        norm = np.abs(arrays[0]).max()
        if norm == 0.:
            return arrays
        return (a / norm for a in arrays)


def test_restart(backend):
    RestartTest(backend=backend).run()
