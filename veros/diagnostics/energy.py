import os

from .diagnostic import VerosDiagnostic
from .. import veros_method
from ..variables import Variable
from ..distributed import global_sum


ENERGY_VARIABLES = dict(
    # mean energy content
    k_m=Variable('Mean kinetic energy', [], 'J', 'Mean kinetic energy',
                 output=True, write_to_restart=True),
    Hd_m=Variable('Mean dynamic enthalpy', [], 'J', 'Mean dynamic enthalpy',
                  output=True, write_to_restart=True),
    eke_m=Variable('Meso-scale eddy energy', [], 'J', 'Meso-scale eddy energy',
                   output=True, write_to_restart=True),
    iw_m=Variable('Internal wave energy', [], 'J', 'Internal wave energy',
                  output=True, write_to_restart=True),
    tke_m=Variable('Turbulent kinetic energy', [], 'J', 'Turbulent kinetic energy',
                   output=True, write_to_restart=True),

    # energy changes
    dE_tot_m=Variable('Change of total energy', [], 'W', 'Change of total energy',
                      output=True, write_to_restart=True),
    dk_m=Variable('Change of KE', [], 'W', 'Change of kinetic energy',
                  output=True, write_to_restart=True),
    dHd_m=Variable('Change of Hd', [], 'W', 'Change of dynamic enthalpy',
                   output=True, write_to_restart=True),
    deke_m=Variable('Change of EKE', [], 'W', 'Change of meso-scale eddy energy',
                    output=True, write_to_restart=True),
    diw_m=Variable('Change of E_iw', [], 'W', 'Change of internal wave energy',
                   output=True, write_to_restart=True),
    dtke_m=Variable('Change of TKE', [], 'W', 'Change of tubulent kinetic energy',
                    output=True, write_to_restart=True),

    # dissipation
    ke_diss_m=Variable('Dissipation of KE', [], 'W', 'Dissipation of kinetic energy',
                       output=True, write_to_restart=True),
    Hd_diss_m=Variable('Dissipation of Hd', [], 'W', 'Dissipation of dynamic enthalpy',
                       output=True, write_to_restart=True),
    eke_diss_m=Variable('Dissipation of EKE', [], 'W', 'Dissipation of meso-scale eddy energy',
                        output=True, write_to_restart=True),
    iw_diss_m=Variable('Dissipation of E_iw', [], 'W', 'Dissipation of internal wave energy',
                       output=True, write_to_restart=True),
    tke_diss_m=Variable('Dissipation of TKE', [], 'W', 'Dissipation of turbulent kinetic energy',
                        output=True, write_to_restart=True),
    adv_diss_m=Variable('Dissipation by advection', [], 'W', 'Dissipation by advection',
                        output=True, write_to_restart=True),

    # external forcing
    wind_m=Variable('Wind work', [], 'W', 'Wind work',
                    output=True, write_to_restart=True),
    dHd_sources_m=Variable('Hd production by ext. sources', [], 'W',
                           'Dynamic enthalpy production through external sources',
                           output=True, write_to_restart=True),
    iw_forc_m=Variable('External forcing of E_iw', [], 'W',
                       'External forcing of internal wave energy',
                       output=True, write_to_restart=True),
    tke_forc_m=Variable('External forcing of TKE', [], 'W',
                        'External forcing of turbulent kinetic energy',
                        output=True, write_to_restart=True),

    # exchange
    ke_hd_m=Variable('Exchange KE -> Hd', [], 'W',
                     'Exchange between kinetic energy and dynamic enthalpy',
                     output=True, write_to_restart=True),
    ke_tke_m=Variable('Exchange KE -> TKE by vert. friction', [], 'W',
                      'Exchange between kinetic energy and turbulent kinetic energy by vertical friction',
                      output=True, write_to_restart=True),
    ke_iw_m=Variable('Exchange KE -> IW by bottom friction', [], 'W',
                     'Exchange between kinetic energy and internal wave energy by bottom friction',
                     output=True, write_to_restart=True),
    tke_hd_m=Variable('Exchange TKE -> Hd by vertical mixing', [], 'W',
                      'Exchange between turbulent kinetic energy and dynamic enthalpy by vertical mixing',
                      output=True, write_to_restart=True),
    ke_eke_m=Variable('Exchange KE -> EKE by lateral friction', [], 'W',
                      'Exchange between kinetic energy and eddy kinetic energy by lateral friction',
                      output=True, write_to_restart=True),
    hd_eke_m=Variable('Exchange Hd -> EKE by GM and lateral mixing', [], 'W',
                      'Exchange between dynamic enthalpy and eddy kinetic energy by GM and lateral mixing',
                      output=True, write_to_restart=True),
    eke_tke_m=Variable('Exchange EKE -> TKE', [], 'W',
                       'Exchange between eddy and turbulent kinetic energy',
                       output=True, write_to_restart=True),
    eke_iw_m=Variable('Exchange EKE -> IW', [], 'W',
                      'Exchange between eddy kinetic energy and internal wave energy',
                      output=True, write_to_restart=True),

    # cabbeling
    cabb_m=Variable('Cabbeling by vertical mixing', [], 'W',
                    'Cabbeling by vertical mixing',
                    output=True, write_to_restart=True),
    cabb_iso_m=Variable('Cabbeling by isopycnal mixing', [], 'W',
                        'Cabbeling by isopycnal mixing',
                        output=True, write_to_restart=True),
)


class Energy(VerosDiagnostic):
    """Diagnose globally averaged energy cycle. Also averages energy in time.
    """
    name = 'energy' #:
    output_path = '{identifier}.energy.nc'  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.
    variables = ENERGY_VARIABLES

    @veros_method
    def initialize(self, vs):
        self.nitts = 0
        for var in self.variables.keys():
            setattr(self, var, 0.)

        output_variables = {key: val for key, val in self.variables.items() if val.output}
        self.initialize_output(vs, output_variables)

    @veros_method
    def diagnose(self, vs):
        # changes of dynamic enthalpy
        vol_t = vs.area_t[2:-2, 2:-2, np.newaxis] \
            * vs.dzt[np.newaxis, np.newaxis, :] \
            * vs.maskT[2:-2, 2:-2, :]
        dP_iso = global_sum(vs,
            np.sum(vol_t * vs.grav / vs.rho_0
                         * (-vs.int_drhodT[2:-2, 2:-2, :, vs.tau]
                           * vs.dtemp_iso[2:-2, 2:-2, :]
                           - vs.int_drhodS[2:-2, 2:-2, :, vs.tau]
                           * vs.dsalt_iso[2:-2, 2:-2, :]))
        )
        dP_hmix = global_sum(vs,
            np.sum(vol_t * vs.grav / vs.rho_0
                         * (-vs.int_drhodT[2:-2, 2:-2, :, vs.tau]
                            * vs.dtemp_hmix[2:-2, 2:-2, :]
                            - vs.int_drhodS[2:-2, 2:-2, :, vs.tau]
                            * vs.dsalt_hmix[2:-2, 2:-2, :]))
        )
        dP_vmix = global_sum(vs,
            np.sum(vol_t * vs.grav / vs.rho_0
                         * (-vs.int_drhodT[2:-2, 2:-2, :, vs.tau]
                            * vs.dtemp_vmix[2:-2, 2:-2, :]
                            - vs.int_drhodS[2:-2, 2:-2, :, vs.tau]
                            * vs.dsalt_vmix[2:-2, 2:-2, :]))
        )
        dP_m = global_sum(vs,
            np.sum(vol_t * vs.grav / vs.rho_0
                      * (-vs.int_drhodT[2:-2, 2:-2, :, vs.tau]
                          * vs.dtemp[2:-2, 2:-2, :, vs.tau]
                          - vs.int_drhodS[2:-2, 2:-2, :, vs.tau]
                          * vs.dsalt[2:-2, 2:-2, :, vs.tau]))
        )
        dP_m_all = dP_m + dP_vmix + dP_hmix + dP_iso

        # changes of kinetic energy
        vol_u = vs.area_u[2:-2, 2:-2, np.newaxis] \
            * vs.dzt[np.newaxis, np.newaxis, :]
        vol_v = vs.area_v[2:-2, 2:-2, np.newaxis] \
            * vs.dzt[np.newaxis, np.newaxis, :]
        k_m = global_sum(vs,
            np.sum(vol_t * 0.5 * (0.5 * (vs.u[2:-2, 2:-2, :, vs.tau] ** 2
                                           + vs.u[1:-3, 2:-2, :, vs.tau] ** 2)
                                    + 0.5 * (vs.v[2:-2, 2:-2, :, vs.tau] ** 2)
                                    + vs.v[2:-2, 1:-3, :, vs.tau] ** 2))
        )
        p_m = global_sum(vs, np.sum(vol_t * vs.Hd[2:-2, 2:-2, :, vs.tau]))
        dk_m = global_sum(vs,
            np.sum(vs.u[2:-2, 2:-2, :, vs.tau] * vs.du[2:-2, 2:-2, :, vs.tau] * vol_u
                      + vs.v[2:-2, 2:-2, :, vs.tau]
                      * vs.dv[2:-2, 2:-2, :, vs.tau] * vol_v
                      + vs.u[2:-2, 2:-2, :, vs.tau] * vs.du_mix[2:-2, 2:-2, :] * vol_u
                      + vs.v[2:-2, 2:-2, :, vs.tau] * vs.dv_mix[2:-2, 2:-2, :] * vol_v)
        )

        # K*Nsqr and KE and dyn. enthalpy dissipation
        vol_w = vs.area_t[2:-2, 2:-2, np.newaxis] * vs.dzw[np.newaxis, np.newaxis, :] \
            * vs.maskW[2:-2, 2:-2, :]
        vol_w[:, :, -1] *= 0.5

        def mean_w(var):
            return global_sum(vs, np.sum(var[2:-2, 2:-2, :] * vol_w))

        mdiss_vmix = mean_w(vs.P_diss_v)
        mdiss_nonlin = mean_w(vs.P_diss_nonlin)
        mdiss_adv = mean_w(vs.P_diss_adv)
        mdiss_hmix = mean_w(vs.P_diss_hmix)
        mdiss_iso = mean_w(vs.P_diss_iso)
        mdiss_skew = mean_w(vs.P_diss_skew)
        mdiss_sources = mean_w(vs.P_diss_sources)

        mdiss_h = mean_w(vs.K_diss_h)
        mdiss_v = mean_w(vs.K_diss_v)
        mdiss_gm = mean_w(vs.K_diss_gm)
        mdiss_bot = mean_w(vs.K_diss_bot)

        wrhom = global_sum(vs, 
            np.sum(-vs.area_t[2:-2, 2:-2, np.newaxis] * vs.maskW[2:-2, 2:-2, :-1]
                       * (vs.p_hydro[2:-2, 2:-2, 1:] - vs.p_hydro[2:-2, 2:-2, :-1])
                       * vs.w[2:-2, 2:-2, :-1, vs.tau])
        )

        # wind work
        if vs.pyom_compatibility_mode:
            wind = global_sum(
                vs,
                np.sum(vs.u[2:-2, 2:-2, -1, vs.tau] * vs.surface_taux[2:-2, 2:-2]
                       * vs.maskU[2:-2, 2:-2, -1] * vs.area_u[2:-2, 2:-2]
                       + vs.v[2:-2, 2:-2, -1, vs.tau] * vs.surface_tauy[2:-2, 2:-2]
                       * vs.maskV[2:-2, 2:-2, -1] * vs.area_v[2:-2, 2:-2])
            )
        else:
            wind = global_sum(
                vs,
                np.sum(vs.u[2:-2, 2:-2, -1, vs.tau] * vs.surface_taux[2:-2, 2:-2] / vs.rho_0
                       * vs.maskU[2:-2, 2:-2, -1] * vs.area_u[2:-2, 2:-2]
                       + vs.v[2:-2, 2:-2, -1, vs.tau] * vs.surface_tauy[2:-2, 2:-2] / vs.rho_0
                       * vs.maskV[2:-2, 2:-2, -1] * vs.area_v[2:-2, 2:-2])
            )

        # meso-scale energy
        if vs.enable_eke:
            eke_m = mean_w(vs.eke[..., vs.tau])
            deke_m = global_sum(vs,
                np.sum(vol_w * (vs.eke[2:-2, 2:-2, :, vs.taup1]
                                - vs.eke[2:-2, 2:-2, :, vs.tau])
                       / vs.dt_tracer)
            )
            eke_diss = mean_w(vs.eke_diss_iw)
            eke_diss_tke = mean_w(vs.eke_diss_tke)
        else:
            eke_m = deke_m = eke_diss_tke = 0.
            eke_diss = mdiss_gm + mdiss_h + mdiss_skew
            if not vs.enable_store_cabbeling_heat:
                eke_diss += -mdiss_hmix - mdiss_iso

        # small-scale energy
        if vs.enable_tke:
            dt_tke = vs.dt_mom
            tke_m = mean_w(vs.tke[..., vs.tau])
            dtke_m = mean_w((vs.tke[..., vs.taup1]
                             - vs.tke[..., vs.tau])
                            / dt_tke)
            tke_diss = mean_w(vs.tke_diss)
            tke_forc = global_sum(vs,
                np.sum(vs.area_t[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2, -1]
                              * (vs.forc_tke_surface[2:-2, 2:-2] + vs.tke_surf_corr[2:-2, 2:-2]))
            )
        else:
            tke_m = dtke_m = tke_diss = tke_forc = 0.

        # internal wave energy
        if vs.enable_idemix:
            iw_m = mean_w(vs.E_iw[..., vs.tau])
            diw_m = global_sum(vs,
                np.sum(vol_w * (vs.E_iw[2:-2, 2:-2, :, vs.taup1]
                                    - vs.E_iw[2:-2, 2:-2, :, vs.tau])
                           / vs.dt_tracer)
            )
            iw_diss = mean_w(vs.iw_diss)

            k = np.maximum(1, vs.kbot[2:-2, 2:-2]) - 1
            mask = k[:, :, np.newaxis] == np.arange(vs.nz)[np.newaxis, np.newaxis, :]
            iwforc = global_sum(vs, 
                np.sum(vs.area_t[2:-2, 2:-2]
                            * (vs.forc_iw_surface[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2, -1]
                               + np.sum(mask * vs.forc_iw_bottom[2:-2, 2:-2, np.newaxis]
                                        * vs.maskW[2:-2, 2:-2, :], axis=2)))
            )
        else:
            iw_m = diw_m = iwforc = 0.
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
        if vs.enable_store_bottom_friction_tke:
            self.ke_tke_m += mdiss_bot
        else:
            self.ke_iw_m += mdiss_bot
        self.eke_tke_m += eke_diss_tke
        self.eke_iw_m += eke_diss
        if not vs.enable_store_cabbeling_heat:
            self.hd_eke_m += -mdiss_hmix - mdiss_iso
            self.tke_hd_m += -mdiss_nonlin

        self.cabb_m += mdiss_nonlin
        self.cabb_iso_m += mdiss_hmix + mdiss_iso

        self.nitts += 1

    @veros_method
    def output(self, vs):
        self.nitts = float(self.nitts or 1)
        output_variables = {key: val for key, val in self.variables.items() if val.output}
        output_data = {key: getattr(self, key) * vs.rho_0 / self.nitts
                       for key in output_variables.keys()}
        if not os.path.isfile(self.get_output_file_name(vs)):
            self.initialize_output(vs, output_variables)
        self.write_output(vs, output_variables, output_data)

        for key in output_variables.keys():
            setattr(self, key, 0.)
        self.nitts = 0

    @veros_method
    def read_restart(self, vs, infile):
        attributes, variables = self.read_h5_restart(vs, self.variables, infile)
        if attributes:
            for key, val in attributes.items():
                setattr(self, key, val)

    @veros_method
    def write_restart(self, vs, outfile):
        restart_data = {key: getattr(self, key)
                        for key, val in self.variables.items() if val.write_to_restart}
        restart_data.update({'nitts': self.nitts})
        self.write_h5_restart(vs, restart_data, {}, {}, outfile)
