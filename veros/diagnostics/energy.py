from .diagnostic import VerosDiagnostic
from .. import veros_class_method

class Energy(VerosDiagnostic):
    """ Diagnose globally averaged energy cycle.

    """
    @veros_class_method
    def initialize(self, veros):
        self.nitts = 0
        self.mean_e = np.zeros(50)
        self.mean_dedt = np.zeros(50)
        self.mean_diss = np.zeros(50)
        self.mean_forc = np.zeros(50)
        self.mean_exchg = np.zeros(50)
        self.mean_misc = np.zeros(50)

    @veros_class_method
    def diagnose(self,veros):
        # changes of dynamic enthalpy
        vol_t = veros.area_t[2:-2, 2:-2, np.newaxis] \
              * veros.dzt[np.newaxis, np.newaxis, :] \
              * veros.maskT[2:-2, 2:-2, :]
        dP_iso = np.sum(vol_t * veros.grav / veros.rho_0 \
                       * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau] \
                         * veros.dtemp_iso[2:-2, 2:-2, :] \
                         - veros.int_drhodS[2:-2, 2:-2, :, veros.tau] \
                         * veros.dsalt_iso[2:-2, 2:-2, :]))
        dP_hmix = np.sum(vol_t * veros.grav / veros.rho_0 \
                       * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau] \
                         * veros.dtemp_hmix[2:-2, 2:-2, :] \
                         - veros.int_drhodS[2:-2, 2:-2, :, veros.tau] \
                         * veros.dsalt_hmix[2:-2, 2:-2, :]))
        dP_vmix = np.sum(vol_t * veros.grav / veros.rho_0 \
                       * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau] \
                         * veros.dtemp_vmix[2:-2, 2:-2, :] \
                         - veros.int_drhodS[2:-2, 2:-2, :, veros.tau] \
                         * veros.dsalt_vmix[2:-2, 2:-2, :]))
        dP_m = np.sum(vol_t * veros.grav / veros.rho_0 \
                       * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau] \
                         * veros.dtemp[2:-2, 2:-2, :, veros.tau] \
                         - veros.int_drhodS[2:-2, 2:-2, :, veros.tau] \
                         * veros.dsalt[2:-2, 2:-2, :, veros.tau]))
        dP_m_all = dP_m + dP_vmix + dP_hmix + dP_iso

        # changes of kinetic energy
        vol_u = veros.area_u[2:-2, 2:-2, np.newaxis] \
              * veros.dzt[np.newaxis, np.newaxis, :]
        vol_v = veros.area_v[2:-2, 2:-2, np.newaxis] \
              * veros.dzt[np.newaxis, np.newaxis, :]
        k_m = np.sum(vol_t * 0.5 * (0.5 * (veros.u[2:-2, 2:-2, :, veros.tau] ** 2 \
                                         + veros.u[1:-3, 2:-2, :, veros.tau] ** 2) \
                                  + 0.5 * (veros.v[2:-2, 2:-2, :, veros.tau] ** 2) \
                                         + veros.v[2:-2, 1:-3, :, veros.tau] ** 2))
        p_m = np.sum(vol_t * veros.Hd[2:-2, 2:-2, :, veros.tau])
        dk_m = np.sum(veros.u[2:-2, 2:-2, :, veros.tau] * veros.du[2:-2, 2:-2, :, veros.tau] * vol_u \
                    + veros.v[2:-2, 2:-2, :, veros.tau] * veros.dv[2:-2, 2:-2, :, veros.tau] * vol_v \
                    + veros.u[2:-2, 2:-2, :, veros.tau] * veros.du_mix[2:-2, 2:-2, :, veros.tau] * vol_u \
                    + veros.v[2:-2, 2:-2, :, veros.tau] * veros.dv_mix[2:-2, 2:-2, :, veros.tau] * vol_v)
        corm = np.sum(veros.u[2:-2, 2:-2, :, veros.tau] * veros.du_cor[2:-2, 2:-2, :] * vol_u \
                    + veros.v[2:-2, 2:-2, :, veros.tau] * veros.dv_cor[2:-2, 2:-2, :] * vol_v)
        k_e_adv = np.sum(veros.u[2:-2, 2:-2, :, veros.tau] * veros.du_adv[2:-2, 2:-2, :] \
                            * vol_u * veros.maskU[2:-2, 2:-2, :] \
                       + veros.v[2:-2, 2:-2, :, veros.tau] * veros.dv_adv[2:-2, 2:-2, :] \
                            * vol_v * veros.maskV[2:-2, 2:-2, :])

        # spurious work by surface pressure
        if not veros.enable_streamfunction:
            spm = np.sum(-veros.u[2:-2, 2:-2, :, veros.tau] * vol_u \
                    * (veros.psi[3:-1, 2:-2, veros.tau] - veros.psi[2:-2, 2:-2, veros.tau]) \
                    / (veros.dxu[2:-2, np.newaxis, np.newaxis] * veros.cost[np.newaxis, 2:-2, np.newaxis]) \
                        - veros.v[2:-2, 2:-2, :, veros.tau] * vol_v \
                    * (veros.psi[2:-2, 3:-1, veros.tau] - veros.psi[2:-2, 2:-2, veros.tau]) \
                    / veros.dyu[np.newaxis, 2:-2, np.newaxis])

        # K*Nsqr and KE and dyn. enthalpy dissipation
        vol_w = veros.area_t[2:-2, 2:-2, np.newaxis] * veros.dzw[np.newaxis, np.newaxis, :] \
                * veros.maskW[2:-2, 2:-2, :]
        vol_w[:,:,-1] *= 0.5
        mdiss_vmix = np.sum(veros.P_diss_v[2:-2, 2:-2, :] * vol_w)
        mdiss_nonlin = np.sum(veros.P_diss_nonlin[2:-2, 2:-2, :] * vol_w)
        mdiss_adv = np.sum(veros.P_diss_adv[2:-2, 2:-2, :] * vol_w)
        mdiss_hmix = np.sum(veros.P_diss_hmix[2:-2, 2:-2, :] * vol_w)
        mdiss_comp = np.sum(veros.P_diss_comp[2:-2, 2:-2, :] * vol_w)
        mdiss_iso = np.sum(veros.P_diss_iso[2:-2, 2:-2, :] * vol_w)
        mdiss_skew = np.sum(veros.P_diss_skew[2:-2, 2:-2, :] * vol_w)
        mdiss_sources = np.sum(veros.P_diss_sources[2:-2, 2:-2, :] * vol_w)

        mdiss_h = np.sum(veros.K_diss_h[2:-2, 2:-2. :] * vol_w)
        mdiss_v = np.sum(veros.K_diss_v[2:-2, 2:-2, :] * vol_w)
        mdiss_gm = np.sum(veros.K_diss_gm[2:-2, 2:-2, :] * vol_w)
        mdiss_bot = np.sum(veros.K_diss_bot[2:-2, 2:-2, :] * vol_w)

        wrhom = np.sum(-veros.area_t[2:-2, 2:-2, np.newaxis] * veros.maskW[2:-2, 2:-2, :-1] \
                     * (veros.p_hydro[2:-2, 2:-2, 1:] - veros.p_hydro[2:-2, 2:-2, :-1]) \
                     * veros.w[2:-2, 2:-2, :-1, veros.tau])

        # wind work
        wind = np.sum(veros.u[2:-2, 2:-2, -1, veros.tau] * veros.surface_taux[2:-2, 2:-2] \
                    * veros.maskU[2:-2, 2:-2, -1] * veros.area_u[2:-2, 2:-2]) \
                    + veros.v[2:-2, 2:-2, -1, veros.tau] * veros.surface_tauy[2:-2, 2:-2] \
                    * veros.maskV[2:-2, 2:-2, -1] * veros.area_v[2:-2, 2:-2]

        # internal wave energy
        if veros.enable_idemix:
            iw_m = np.sum(vol_w * veros.E_iw[2:-2, 2:-2, :, veros.tau])
            diw_m = np.sum(vol_w * (veros.E_iw[2:-2, 2:-2, :, veros.taup1] \
                                  - veros.E_iw[2:-2, 2:-2, :, veros.tau]) \
                           / veros.dt_tracer)
            iw_diss = np.sum(veros.iw_diss[2:-2, 2:-2, :])

            k = np.maximum(1, veros.kbot[2:-2, 2:-2]) - 1
            mask = k[:, :, np.newaxis] == np.arange(veros.nz)[np.newaxis, np.newaxis, :]
            iwforc = np.sum(veros.area_t[2:-2, 2:-2] \
                       * (veros.forc_iw_surface[2:-2, 2:-2] * veros.maskW[2:-2, 2:-2, -1] \
                        + np.sum(mask * veros.forc_iw_bottom[2:-2, 2:-2, np.newaxis] \
                                      * veros.maskW[2:-2, 2:-2, :], axis=2)))

        # NIW low mode compartment

    @veros_class_method
    def output(self,veros):
        import warnings
        warnings.warn("routine is not implemented yet")

    @veros_class_method
    def read_restart(self, veros):
        pass

    @veros_class_method
    def write_restart(self, veros):
        pass
