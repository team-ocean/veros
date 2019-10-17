
import pkg_resources

import numpy as np
import h5netcdf

from test_base import VerosPyOMSystemTest, VerosLegacyDummy
from veros import veros_method, tools

DATA_FILES = tools.get_assets('global_4deg', pkg_resources.resource_filename('veros', 'setup/global_4deg/assets.yml'))


class GlobalFourDegreeSetup(VerosLegacyDummy):
    """ global 4 deg model with 15 levels
    """
    def set_parameter(self, vs):
        vs.identifier = '4deg_test'
        vs.diskless_mode = True
        vs.pyom_compatibility_mode = True

        M = self.main_module

        (M.nx, M.ny, M.nz) = (90, 40, 15)
        M.dt_mom = 1800.0
        M.dt_tracer = 86400.0
        M.AB_eps = 0.1

        M.coord_degree = True
        M.enable_cyclic_x = True

        M.congr_epsilon = 1e-8
        M.congr_max_iterations = 20000

        I = self.isoneutral_module
        I.enable_neutral_diffusion = True
        I.K_iso_0 = 1000.0
        I.K_iso_steep = 1000.0
        I.iso_dslope = 4. / 1000.0
        I.iso_slopec = 1. / 1000.0
        I.enable_skew_diffusion = True

        M.enable_hor_friction = True
        M.A_h = 1.75e6
        M.enable_hor_friction_cos_scaling = True
        M.hor_friction_cosPower = 1

        M.enable_implicit_vert_friction = True
        T = self.tke_module
        T.enable_tke = True
        T.c_k = 0.1
        T.c_eps = 0.7
        T.alpha_tke = 30.0
        T.mxl_min = 1e-8
        T.tke_mxl_choice = 2
        T.enable_tke_superbee_advection = True

        E = self.eke_module
        E.enable_eke = True
        E.eke_k_max = 1e4
        E.eke_c_k = 0.4
        E.eke_c_eps = 0.5
        E.eke_cross = 2.
        E.eke_crhin = 1.0
        E.eke_lmin = 100.0
        E.eke_int_diss0 = 5.78703703704e-07
        E.kappa_EKE0 = 0.1
        E.enable_eke_superbee_advection = True

        I = self.idemix_module
        I.enable_idemix = True
        I.gamma = 1.57
        I.mu0 = 1.33333333333
        I.enable_idemix_hor_diffusion = True
        I.enable_eke_diss_surfbot = True
        I.eke_diss_surfbot_frac = 0.2 # fraction which goes into bottom
        I.enable_idemix_superbee_advection = True
        I.tau_v = 86400.
        I.jstar = 10.

        M.eq_of_state_type = 5

    @veros_method
    def _read_forcing(self, vs, var):
        with h5netcdf.File(DATA_FILES['forcing'], 'r') as infile:
            return np.array(infile.variables[var][...].T.astype(str(infile.variables[var].dtype)))

    @veros_method
    def set_grid(self, vs):

        m = self.main_module
        ddz = np.array([50., 70., 100., 140., 190., 240., 290., 340.,
                        390., 440., 490., 540., 590., 640., 690.])
        m.dzt[:] = ddz[::-1]
        m.dxt[:] = 4.0
        m.dyt[:] = 4.0
        m.y_origin = -76.0
        m.x_origin = 4.0

    @veros_method
    def set_coriolis(self, vs):
        m = self.main_module
        m.coriolis_t[...] = 2 * m.omega * np.sin(m.yt[np.newaxis, :] / 180. * m.pi)

    @veros_method
    def set_topography(self, vs):
        m = self.main_module
        bathymetry_data = self._read_forcing(vs, 'bathymetry')
        salt_data = self._read_forcing(vs, 'salinity')[:, :, ::-1]
        mask_salt = salt_data == 0.
        m.kbot[2:-2, 2:-2] = 1 + np.sum(mask_salt.astype(np.int), axis=2)
        mask_bathy = bathymetry_data == 0
        m.kbot[2:-2, 2:-2][mask_bathy] = 0
        m.kbot[m.kbot == m.nz] = 0

    @veros_method
    def set_initial_conditions(self, vs):
        m = self.main_module

        self.taux, self.tauy, self.qnec, self.qnet, self.sss_clim, self.sst_clim = (
            np.zeros((m.nx + 4, m.ny + 4, 12)) for _ in range(6))

        # initial conditions for T and S
        temp_data = self._read_forcing(vs, 'temperature')[:, :, ::-1]
        m.temp[2:-2, 2:-2, :, :] = (temp_data[:, :, :, np.newaxis]
                                    * m.maskT[2:-2, 2:-2, :, np.newaxis])

        salt_data = self._read_forcing(vs, 'salinity')[:, :, ::-1]
        m.salt[2:-2, 2:-2, :, :] = (salt_data[..., np.newaxis]
                                    * m.maskT[2:-2, 2:-2, :, np.newaxis])

        # use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
        self.taux[2:-2, 2:-2, :] = self._read_forcing(vs, 'tau_x') / m.rho_0
        self.tauy[2:-2, 2:-2, :] = self._read_forcing(vs, 'tau_y') / m.rho_0

        # heat flux
        with h5netcdf.File(DATA_FILES['ecmwf'], 'r') as ecmwf_data:
            self.qnec[2:-2, 2:-2, :] = np.array(ecmwf_data.variables['Q3'][...].astype('float64')).T
            self.qnec[self.qnec <= -1e10] = 0.0

        q = self._read_forcing(vs, 'q_net')
        self.qnet[2:-2, 2:-2, :] = -q
        self.qnet[self.qnet <= -1e10] = 0.0

        fxa = np.sum(self.qnet[2:-2, 2:-2, :] * m.area_t[2:-2, 2:-2, np.newaxis])
        fxb = 12. * np.sum(m.area_t[2:-2, 2:-2])

        fxa = float(fxa / fxb)
        try:
            maskT = m.maskT[:, :, -1, np.newaxis].copy2numpy()
        except AttributeError:
            maskT = m.maskT[:, :, -1, np.newaxis]
        self.qnet[...] = (self.qnet - fxa) * maskT

        # SST and SSS
        self.sst_clim[2:-2, 2:-2, :] = self._read_forcing(vs, 'sst')
        self.sss_clim[2:-2, 2:-2, :] = self._read_forcing(vs, 'sss')

        idm = self.idemix_module
        if idm.enable_idemix:
            idm.forc_iw_bottom[2:-2, 2:-2] = self._read_forcing(vs, 'tidal_energy') / m.rho_0
            idm.forc_iw_surface[2:-2, 2:-2] = self._read_forcing(vs, 'wind_energy') / m.rho_0 * 0.2

    @veros_method
    def set_forcing(self, vs):
        m = self.main_module

        year_in_seconds = 360 * 86400.
        (n1, f1), (n2, f2) = tools.get_periodic_interval(vs.time, year_in_seconds,
                                                         year_in_seconds / 12., 12)

        # wind stress
        m.surface_taux[...] = (f1 * self.taux[:, :, n1] + f2 * self.taux[:, :, n2])
        m.surface_tauy[...] = (f1 * self.tauy[:, :, n1] + f2 * self.tauy[:, :, n2])

        # tke flux
        t = self.tke_module
        if t.enable_tke:
            t.forc_tke_surface[1:-1, 1:-1] = np.sqrt((0.5 * (m.surface_taux[1:-1, 1:-1] \
                                                                + m.surface_taux[:-2, 1:-1]))**2
                                                      + (0.5 * (m.surface_tauy[1:-1, 1:-1] \
                                                                + m.surface_tauy[1:-1, :-2]))**2)**(3. / 2.)
        # heat flux : W/m^2 K kg/J m^3/kg = K m/s
        cp_0 = 3991.86795711963
        sst = f1 * self.sst_clim[:, :, n1] + f2 * self.sst_clim[:, :, n2]
        qnec = f1 * self.qnec[:, :, n1] + f2 * self.qnec[:, :, n2]
        qnet = f1 * self.qnet[:, :, n1] + f2 * self.qnet[:, :, n2]
        m.forc_temp_surface[...] = (qnet + qnec * (sst - m.temp[:, :, -1, self.get_tau()])) \
                                    * m.maskT[:, :, -1] / cp_0 / m.rho_0

        # salinity restoring
        t_rest = 30 * 86400.0
        sss = f1 * self.sss_clim[:, :, n1] + f2 * self.sss_clim[:, :, n2]
        m.forc_salt_surface[:] = 1. / t_rest * \
            (sss - m.salt[:, :, -1, self.get_tau()]) * m.maskT[:, :, -1] * m.dzt[-1]

        # apply simple ice mask
        mask = np.logical_and(m.temp[:, :, -1, self.get_tau()] * m.maskT[:, :, -1] < -1.8,
                              m.forc_temp_surface < 0.)
        m.forc_temp_surface[mask] = 0.0
        m.forc_salt_surface[mask] = 0.0

        if m.enable_tempsalt_sources:
            m.temp_source[:] = m.maskT * self.rest_tscl * \
                (f1 * self.t_star[:, :, :, n1] + f2 * self.t_star[:, :, :, n2] \
                - m.temp[:, :, :, self.get_tau()])
            m.salt_source[:] = self.maskT * self.rest_tscl * \
                (f1 * self.s_star[:, :, :, n1] + f2 * self.s_star[:, :, :, n2] \
                - m.salt[:, :, :, self.get_tau()])

    def set_diagnostics(self, vs):
        pass

    def after_timestep(self, vs):
        pass


class FourDegreeTest(VerosPyOMSystemTest):
    Testclass = GlobalFourDegreeSetup
    timesteps = 100

    def test_passed(self):
        differing_scalars = self.check_scalar_objects()
        differing_arrays = self.check_array_objects()

        if differing_scalars or differing_arrays:
            print('The following attributes do not match between old and new veros:')
            for s, (v1, v2) in differing_scalars.items():
                print('{}, {}, {}'.format(s, v1, v2))
            for a, (v1, v2) in differing_arrays.items():
                if a in ('Ai_ez', 'Ai_nz', 'Ai_bx', 'Ai_by'):
                    # usually very small differences being amplified
                    continue
                self.check_variable(a, atol=1e-4, data=(v1, v2))


def test_4deg(pyom2_lib, backend):
    FourDegreeTest(fortran=pyom2_lib, backend=backend).run()
