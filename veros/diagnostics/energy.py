import os

from .diagnostic import VerosDiagnostic
from .. import veros_class_method
from ..variables import Variable

ENERGY_VARIABLES = dict(
    # mean energy content
    k_m=Variable("Mean kinetic energy", [], "J", "Mean kinetic energy",
                 output=True, write_to_restart=True),
    Hd_m=Variable("Mean dynamic enthalpy", [], "J", "Mean dynamic enthalpy",
                  output=True, write_to_restart=True),
    eke_m=Variable("Meso-scale eddy energy", [], "J", "Meso-scale eddy energy",
                   output=True, write_to_restart=True),
    iw_m=Variable("Internal wave energy", [], "J", "Internal wave energy",
                  output=True, write_to_restart=True),
    tke_m=Variable("Turbulent kinetic energy", [], "J", "Turbulent kinetic energy",
                   output=True, write_to_restart=True),

    # energy changes
    dE_tot_m=Variable("Change of total energy", [], "W", "Change of total energy",
                      output=True, write_to_restart=True),
    dk_m=Variable("Change of KE", [], "W", "Change of kinetic energy",
                  output=True, write_to_restart=True),
    dHd_m=Variable("Change of Hd", [], "W", "Change of dynamic enthalpy",
                   output=True, write_to_restart=True),
    deke_m=Variable("Change of EKE", [], "W", "Change of meso-scale eddy energy",
                    output=True, write_to_restart=True),
    diw_m=Variable("Change of E_iw", [], "W", "Change of internal wave energy",
                   output=True, write_to_restart=True),
    dtke_m=Variable("Change of TKE", [], "W", "Change of tubulent kinetic energy",
                    output=True, write_to_restart=True),

    # dissipation
    ke_diss_m=Variable("Dissipation of KE", [], "W", "Dissipation of kinetic energy",
                       output=True, write_to_restart=True),
    Hd_diss_m=Variable("Dissipation of Hd", [], "W", "Dissipation of dynamic enthalpy",
                       output=True, write_to_restart=True),
    eke_diss_m=Variable("Dissipation of EKE", [], "W", "Dissipation of meso-scale eddy energy",
                        output=True, write_to_restart=True),
    iw_diss_m=Variable("Dissipation of E_iw", [], "W", "Dissipation of internal wave energy",
                       output=True, write_to_restart=True),
    tke_diss_m=Variable("Dissipation of TKE", [], "W", "Dissipation of turbulent kinetic energy",
                        output=True, write_to_restart=True),
    adv_diss_m=Variable("Dissipation by advection", [], "W", "Dissipation by advection",
                        output=True, write_to_restart=True),

    # external forcing
    wind_m=Variable("Wind work", [], "W", "Wind work",
                    output=True, write_to_restart=True),
    dHd_sources_m=Variable("Hd production by ext. sources", [], "W",
                           "Dynamic enthalpy production through external sources",
                           output=True, write_to_restart=True),
    iw_forc_m=Variable("External forcing of E_iw", [], "W",
                       "External forcing of internal wave energy",
                       output=True, write_to_restart=True),
    tke_forc_m=Variable("External forcing of TKE", [], "W",
                        "External forcing of turbulent kinetic energy",
                        output=True, write_to_restart=True),

    # exchange
    ke_hd_m=Variable("Exchange KE -> Hd", [], "W",
                     "Exchange between kinetic energy and dynamic enthalpy",
                     output=True, write_to_restart=True),
    ke_tke_m=Variable("Exchange KE -> TKE by vert. friction", [], "W",
                      "Exchange between kinetic energy and turbulent kinetic energy by vertical friction",
                      output=True, write_to_restart=True),
    ke_iw_m=Variable("Exchange KE -> IW by bottom friction", [], "W",
                     "Exchange between kinetic energy and internal wave energy by bottom friction",
                     output=True, write_to_restart=True),
    tke_hd_m=Variable("Exchange TKE -> Hd by vertical mixing", [], "W",
                      "Exchange between turbulent kinetic energy and dynamic enthalpy by vertical mixing",
                      output=True, write_to_restart=True),
    ke_eke_m=Variable("Exchange KE -> EKE by lateral friction", [], "W",
                      "Exchange between kinetic energy and eddy kinetic energy by lateral friction",
                      output=True, write_to_restart=True),
    hd_eke_m=Variable("Exchange Hd -> EKE by GM and lateral mixing", [], "W",
                      "Exchange between dynamic enthalpy and eddy kinetic energy by GM and lateral mixing",
                      output=True, write_to_restart=True),
    eke_tke_m=Variable("Exchange EKE -> TKE", [], "W",
                       "Exchange between eddy and turbulent kinetic energy",
                       output=True, write_to_restart=True),
    eke_iw_m=Variable("Exchange EKE -> IW", [], "W",
                      "Exchange between eddy kinetic energy and internal wave energy",
                      output=True, write_to_restart=True),

    # cabbeling
    cabb_m=Variable("Cabbeling by vertical mixing", [], "W",
                    "Cabbeling by vertical mixing",
                    output=True, write_to_restart=True),
    cabb_iso_m=Variable("Cabbeling by isopycnal mixing", [], "W",
                        "Cabbeling by isopycnal mixing",
                        output=True, write_to_restart=True),
)


class Energy(VerosDiagnostic):
    """Diagnose globally averaged energy cycle. Also averages energy in time.
    """
    output_path = "{identifier}_energy.nc"  # : File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  # : Frequency (in seconds) in which output is written.
    sampling_frequency = None  # : Frequency (in seconds) in which variables are accumulated.
    variables = ENERGY_VARIABLES

    @veros_class_method
    def initialize(self, veros):
        self.nitts = 0
        for var in self.variables.keys():
            setattr(self, var, 0.)

        output_variables = {key: val for key, val in self.variables.items() if val.output}
        self.initialize_output(veros, output_variables)

    @veros_class_method
    def diagnose(self, veros):
        # changes of dynamic enthalpy
        vol_t = veros.area_t[2:-2, 2:-2, np.newaxis] \
            * veros.dzt[np.newaxis, np.newaxis, :] \
            * veros.maskT[2:-2, 2:-2, :]
        dP_iso = np.sum(vol_t * veros.grav / veros.rho_0
                        * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau]
                           * veros.dtemp_iso[2:-2, 2:-2, :]
                           - veros.int_drhodS[2:-2, 2:-2, :, veros.tau]
                           * veros.dsalt_iso[2:-2, 2:-2, :]))
        dP_hmix = np.sum(vol_t * veros.grav / veros.rho_0
                         * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau]
                            * veros.dtemp_hmix[2:-2, 2:-2, :]
                            - veros.int_drhodS[2:-2, 2:-2, :, veros.tau]
                            * veros.dsalt_hmix[2:-2, 2:-2, :]))
        dP_vmix = np.sum(vol_t * veros.grav / veros.rho_0
                         * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau]
                            * veros.dtemp_vmix[2:-2, 2:-2, :]
                            - veros.int_drhodS[2:-2, 2:-2, :, veros.tau]
                            * veros.dsalt_vmix[2:-2, 2:-2, :]))
        dP_m = np.sum(vol_t * veros.grav / veros.rho_0
                      * (-veros.int_drhodT[2:-2, 2:-2, :, veros.tau]
                          * veros.dtemp[2:-2, 2:-2, :, veros.tau]
                          - veros.int_drhodS[2:-2, 2:-2, :, veros.tau]
                          * veros.dsalt[2:-2, 2:-2, :, veros.tau]))
        dP_m_all = dP_m + dP_vmix + dP_hmix + dP_iso

        # changes of kinetic energy
        vol_u = veros.area_u[2:-2, 2:-2, np.newaxis] \
            * veros.dzt[np.newaxis, np.newaxis, :]
        vol_v = veros.area_v[2:-2, 2:-2, np.newaxis] \
            * veros.dzt[np.newaxis, np.newaxis, :]
        k_m = np.sum(vol_t * 0.5 * (0.5 * (veros.u[2:-2, 2:-2, :, veros.tau] ** 2
                                           + veros.u[1:-3, 2:-2, :, veros.tau] ** 2)
                                    + 0.5 * (veros.v[2:-2, 2:-2, :, veros.tau] ** 2)
                                    + veros.v[2:-2, 1:-3, :, veros.tau] ** 2))
        p_m = np.sum(vol_t * veros.Hd[2:-2, 2:-2, :, veros.tau])
        dk_m = np.sum(veros.u[2:-2, 2:-2, :, veros.tau] * veros.du[2:-2, 2:-2, :, veros.tau] * vol_u
                      + veros.v[2:-2, 2:-2, :, veros.tau]
                      * veros.dv[2:-2, 2:-2, :, veros.tau] * vol_v
                      + veros.u[2:-2, 2:-2, :, veros.tau] * veros.du_mix[2:-2, 2:-2, :] * vol_u
                      + veros.v[2:-2, 2:-2, :, veros.tau] * veros.dv_mix[2:-2, 2:-2, :] * vol_v)

        corm = np.sum(veros.u[2:-2, 2:-2, :, veros.tau] * veros.du_cor[2:-2, 2:-2, :] * vol_u
                      + veros.v[2:-2, 2:-2, :, veros.tau] * veros.dv_cor[2:-2, 2:-2, :] * vol_v)
        k_e_adv = np.sum(veros.u[2:-2, 2:-2, :, veros.tau] * veros.du_adv[2:-2, 2:-2, :]
                         * vol_u * veros.maskU[2:-2, 2:-2, :]
                         + veros.v[2:-2, 2:-2, :, veros.tau] * veros.dv_adv[2:-2, 2:-2, :]
                         * vol_v * veros.maskV[2:-2, 2:-2, :])

        # spurious work by surface pressure
        if not veros.enable_streamfunction:
            spm = np.sum(-veros.u[2:-2, 2:-2, :, veros.tau] * vol_u
                         * (veros.psi[3:-1, 2:-2, veros.tau] - veros.psi[2:-2, 2:-2, veros.tau])
                         / (veros.dxu[2:-2, np.newaxis, np.newaxis] * veros.cost[np.newaxis, 2:-2, np.newaxis])
                         - veros.v[2:-2, 2:-2, :, veros.tau] * vol_v
                         * (veros.psi[2:-2, 3:-1, veros.tau] - veros.psi[2:-2, 2:-2, veros.tau])
                         / veros.dyu[np.newaxis, 2:-2, np.newaxis])

        # K*Nsqr and KE and dyn. enthalpy dissipation
        vol_w = veros.area_t[2:-2, 2:-2, np.newaxis] * veros.dzw[np.newaxis, np.newaxis, :] \
            * veros.maskW[2:-2, 2:-2, :]
        vol_w[:, :, -1] *= 0.5

        def mean_w(var):
            return np.sum(var[2:-2, 2:-2, :] * vol_w)

        mdiss_vmix = mean_w(veros.P_diss_v)
        mdiss_nonlin = mean_w(veros.P_diss_nonlin)
        mdiss_adv = mean_w(veros.P_diss_adv)
        mdiss_hmix = mean_w(veros.P_diss_hmix)
        mdiss_comp = mean_w(veros.P_diss_comp)
        mdiss_iso = mean_w(veros.P_diss_iso)
        mdiss_skew = mean_w(veros.P_diss_skew)
        mdiss_sources = mean_w(veros.P_diss_sources)

        mdiss_h = mean_w(veros.K_diss_h)
        mdiss_v = mean_w(veros.K_diss_v)
        mdiss_gm = mean_w(veros.K_diss_gm)
        mdiss_bot = mean_w(veros.K_diss_bot)

        wrhom = np.sum(-veros.area_t[2:-2, 2:-2, np.newaxis] * veros.maskW[2:-2, 2:-2, :-1]
                       * (veros.p_hydro[2:-2, 2:-2, 1:] - veros.p_hydro[2:-2, 2:-2, :-1])
                       * veros.w[2:-2, 2:-2, :-1, veros.tau])

        # wind work
        wind = np.sum(veros.u[2:-2, 2:-2, -1, veros.tau] * veros.surface_taux[2:-2, 2:-2]
                      * veros.maskU[2:-2, 2:-2, -1] * veros.area_u[2:-2, 2:-2]
                      + veros.v[2:-2, 2:-2, -1, veros.tau] * veros.surface_tauy[2:-2, 2:-2]
                      * veros.maskV[2:-2, 2:-2, -1] * veros.area_v[2:-2, 2:-2])

        # internal wave energy
        if veros.enable_idemix:
            iw_m = mean_w(veros.E_iw[..., veros.tau])
            diw_m = np.sum(vol_w * (veros.E_iw[2:-2, 2:-2, :, veros.taup1]
                                    - veros.E_iw[2:-2, 2:-2, :, veros.tau])
                           / veros.dt_tracer)
            iw_diss = mean_w(veros.iw_diss)

            k = np.maximum(1, veros.kbot[2:-2, 2:-2]) - 1
            mask = k[:, :, np.newaxis] == np.arange(veros.nz)[np.newaxis, np.newaxis, :]
            iwforc = np.sum(veros.area_t[2:-2, 2:-2]
                            * (veros.forc_iw_surface[2:-2, 2:-2] * veros.maskW[2:-2, 2:-2, -1]
                               + np.sum(mask * veros.forc_iw_bottom[2:-2, 2:-2, np.newaxis]
                                        * veros.maskW[2:-2, 2:-2, :], axis=2)))

        # meso-scale energy
        if veros.enable_eke:
            eke_m = mean_w(veros.eke[..., veros.tau])
            deke_m = np.sum(vol_w * (veros.eke[2:-2, 2:-2, :, veros.taup1]
                                     - veros.eke[2:-2, 2:-2, :, veros.tau])
                            / veros.dt_tracer)
            eke_diss = mean_w(veros.eke_diss_iw)
            eke_diss_tke = mean_w(veros.eke_diss_tke)

        # small-scale energy
        if veros.enable_tke:
            tke_m = mean_w(veros.tke[..., veros.tau])
            dtke_m = mean_w((veros.tke[..., veros.taup1]
                             - veros.tke[..., veros.tau])
                            / veros.dt_tke)
            tke_diss = mean_w(veros.tke_diss)
            tke_forc = np.sum(veros.area_t[2:-2, 2:-2] * veros.maskW[2:-2, 2:-2, -1]
                              * (veros.forc_tke_surface[2:-2, 2:-2] + veros.tke_surf_corr[2:-2, 2:-2]))

        # shortcut for EKE model
        if not veros.enable_eke:
            eke_diss = mdiss_gm + mdiss_h + mdiss_skew
            if not veros.enable_store_cabbeling_heat:
                eke_diss += -mdiss_hmix - mdiss_iso

        # shortcut for IW model
        if not veros.enable_idemix:
            iw_diss = eke_diss

        # store results
        self.k_m += k_m
        self.Hd_m += p_m
        self.eke_m += eke_m
        self.iw_m += iw_m
        self.tke_m += tke_m

        self.dk_m += dk_m
        self.dHd_m += dP_m_all + mdiss_sources
        self.deke_m += deke_m
        self.diw_m += diw_m
        self.dtke_m += dtke_m
        self.dE_tot_m += self.dk_m + self.dHd_m + self.deke_m + self.diw_m + self.dtke_m

        self.wind_m += wind
        self.dHd_sources_m += mdiss_sources
        self.iw_forc_m += iwforc
        self.tke_forc_m += tke_forc

        self.ke_diss_m += mdiss_h + mdiss_v + mdiss_gm + mdiss_bot
        self.Hd_diss_m += mdiss_vmix + mdiss_nonlin + mdiss_hmix + mdiss_adv \
            + mdiss_iso + mdiss_skew
        self.eke_diss_m += eke_diss + eke_diss_tke
        self.iw_diss_m += iw_diss
        self.tke_diss_m += tke_diss
        self.adv_diss_m += mdiss_adv

        self.ke_hd_m += wrhom
        self.ke_eke_m += mdiss_h + mdiss_gm
        self.hd_eke_m += -mdiss_skew
        self.ke_tke_m += mdiss_v
        self.tke_hd_m += -mdiss_vmix - mdiss_adv
        if veros.enable_store_bottom_friction_tke:
            self.ke_tke_m += mdiss_bot
        else:
            self.ke_iw_m += mdiss_bot
        self.eke_tke_m += eke_diss_tke
        self.eke_iw_m += eke_diss
        if not veros.enable_store_cabbeling_heat:
            self.hd_eke_m += -mdiss_hmix - mdiss_iso
            self.tke_hd_m += -mdiss_nonlin

        self.cabb_m += mdiss_nonlin
        self.cabb_iso_m += mdiss_hmix + mdiss_iso

        self.nitts += 1

    @veros_class_method
    def output(self, veros):
        self.nitts = float(self.nitts or 1)
        output_variables = {key: val for key, val in self.variables.items() if val.output}
        output_data = {key: getattr(self, key) * veros.rho_0 /
                       self.nitts for key in output_variables.keys()}
        if not os.path.isfile(self.get_output_file_name(veros)):
            self.initialize_output(veros, output_variables)
        self.write_output(veros, output_variables, output_data)

        for key in output_variables.keys():
            setattr(self, key, 0.)
        self.nitts = 0

    @veros_class_method
    def read_restart(self, veros):
        attributes, variables = self.read_h5_restart(veros)
        if attributes:
            for key, val in attributes.items():
                setattr(self, key, val)

    @veros_class_method
    def write_restart(self, veros):
        restart_data = {key: getattr(self, key)
                        for key, val in self.variables.items() if val.write_to_restart}
        restart_data.update({"nitts": self.nitts})
        self.write_h5_restart(veros, restart_data, {}, {})
