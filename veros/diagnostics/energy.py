import os

from veros import veros_kernel, KernelOutput, runtime_settings
from veros.core.operators import numpy as npx, update_multiply, at
from veros.diagnostics.base import VerosDiagnostic
from veros.variables import Variable
from veros.distributed import global_sum


ENERGY_VARIABLES = dict(
    nitts=Variable("nitts", None, write_to_restart=True),
    # mean energy content
    k_m=Variable("Mean kinetic energy", None, "J", "Mean kinetic energy", write_to_restart=True),
    Hd_m=Variable("Mean dynamic enthalpy", None, "J", "Mean dynamic enthalpy", write_to_restart=True),
    eke_m=Variable("Meso-scale eddy energy", None, "J", "Meso-scale eddy energy", write_to_restart=True),
    iw_m=Variable("Internal wave energy", None, "J", "Internal wave energy", write_to_restart=True),
    tke_m=Variable("Turbulent kinetic energy", None, "J", "Turbulent kinetic energy", write_to_restart=True),
    # energy changes
    dE_tot_m=Variable("Change of total energy", None, "W", "Change of total energy", write_to_restart=True),
    dk_m=Variable("Change of KE", None, "W", "Change of kinetic energy", write_to_restart=True),
    dHd_m=Variable("Change of Hd", None, "W", "Change of dynamic enthalpy", write_to_restart=True),
    deke_m=Variable("Change of EKE", None, "W", "Change of meso-scale eddy energy", write_to_restart=True),
    diw_m=Variable("Change of E_iw", None, "W", "Change of internal wave energy", write_to_restart=True),
    dtke_m=Variable("Change of TKE", None, "W", "Change of tubulent kinetic energy", write_to_restart=True),
    # dissipation
    ke_diss_m=Variable("Dissipation of KE", None, "W", "Dissipation of kinetic energy", write_to_restart=True),
    Hd_diss_m=Variable("Dissipation of Hd", None, "W", "Dissipation of dynamic enthalpy", write_to_restart=True),
    eke_diss_m=Variable(
        "Dissipation of EKE", None, "W", "Dissipation of meso-scale eddy energy", write_to_restart=True
    ),
    iw_diss_m=Variable("Dissipation of E_iw", None, "W", "Dissipation of internal wave energy", write_to_restart=True),
    tke_diss_m=Variable(
        "Dissipation of TKE", None, "W", "Dissipation of turbulent kinetic energy", write_to_restart=True
    ),
    adv_diss_m=Variable("Dissipation by advection", None, "W", "Dissipation by advection", write_to_restart=True),
    # external forcing
    wind_m=Variable("Wind work", None, "W", "Wind work", write_to_restart=True),
    dHd_sources_m=Variable(
        "Hd production by ext. sources",
        None,
        "W",
        "Dynamic enthalpy production through external sources",
        write_to_restart=True,
    ),
    iw_forc_m=Variable(
        "External forcing of E_iw", None, "W", "External forcing of internal wave energy", write_to_restart=True
    ),
    tke_forc_m=Variable(
        "External forcing of TKE", None, "W", "External forcing of turbulent kinetic energy", write_to_restart=True
    ),
    # exchange
    ke_hd_m=Variable(
        "Exchange KE -> Hd", None, "W", "Exchange between kinetic energy and dynamic enthalpy", write_to_restart=True
    ),
    ke_tke_m=Variable(
        "Exchange KE -> TKE by vert. friction",
        None,
        "W",
        "Exchange between kinetic energy and turbulent kinetic energy by vertical friction",
        write_to_restart=True,
    ),
    ke_iw_m=Variable(
        "Exchange KE -> IW by bottom friction",
        None,
        "W",
        "Exchange between kinetic energy and internal wave energy by bottom friction",
        write_to_restart=True,
    ),
    tke_hd_m=Variable(
        "Exchange TKE -> Hd by vertical mixing",
        None,
        "W",
        "Exchange between turbulent kinetic energy and dynamic enthalpy by vertical mixing",
        write_to_restart=True,
    ),
    ke_eke_m=Variable(
        "Exchange KE -> EKE by lateral friction",
        None,
        "W",
        "Exchange between kinetic energy and eddy kinetic energy by lateral friction",
        write_to_restart=True,
    ),
    hd_eke_m=Variable(
        "Exchange Hd -> EKE by GM and lateral mixing",
        None,
        "W",
        "Exchange between dynamic enthalpy and eddy kinetic energy by GM and lateral mixing",
        write_to_restart=True,
    ),
    eke_tke_m=Variable(
        "Exchange EKE -> TKE", None, "W", "Exchange between eddy and turbulent kinetic energy", write_to_restart=True
    ),
    eke_iw_m=Variable(
        "Exchange EKE -> IW",
        None,
        "W",
        "Exchange between eddy kinetic energy and internal wave energy",
        write_to_restart=True,
    ),
    # cabbeling
    cabb_m=Variable("Cabbeling by vertical mixing", None, "W", "Cabbeling by vertical mixing", write_to_restart=True),
    cabb_iso_m=Variable(
        "Cabbeling by isopycnal mixing", None, "W", "Cabbeling by isopycnal mixing", write_to_restart=True
    ),
)


DEFAULT_OUTPUT_VARS = [var for var in ENERGY_VARIABLES.keys() if var not in ("nitts",)]


class Energy(VerosDiagnostic):
    """Diagnose globally averaged energy cycle. Also averages energy in time."""

    name = "energy"  #:
    output_path = "{identifier}.energy.nc"  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.

    var_meta = ENERGY_VARIABLES

    def __init__(self, state):
        self.output_variables = DEFAULT_OUTPUT_VARS.copy()

    def initialize(self, state):
        self.initialize_variables(state)
        self.initialize_output(state)

    def diagnose(self, state):
        energies = diagnose_kernel(state)

        # store results
        for energy, val in energies._asdict().items():
            total_val = self.variables.get(energy)
            setattr(self.variables, energy, total_val + val)

        self.variables.nitts = self.variables.nitts + 1

    def output(self, state):
        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        energy_vs = self.variables
        nitts = float(energy_vs.nitts or 1)

        for key in self.output_variables:
            val = getattr(energy_vs, key)
            setattr(energy_vs, key, val * state.settings.rho_0 / nitts)

        self.write_output(state)

        for key in self.output_variables:
            setattr(energy_vs, key, 0.0)

        energy_vs.nitts = 0


@veros_kernel
def diagnose_kernel(state):
    vs = state.variables
    settings = state.settings

    # changes of dynamic enthalpy
    vol_t = vs.area_t[2:-2, 2:-2, npx.newaxis] * vs.dzt[npx.newaxis, npx.newaxis, :] * vs.maskT[2:-2, 2:-2, :]

    dP_iso = global_sum(
        npx.sum(
            vol_t
            * settings.grav
            / settings.rho_0
            * (
                -vs.int_drhodT[2:-2, 2:-2, :, vs.tau] * vs.dtemp_iso[2:-2, 2:-2, :]
                - vs.int_drhodS[2:-2, 2:-2, :, vs.tau] * vs.dsalt_iso[2:-2, 2:-2, :]
            )
        )
    )

    dP_hmix = global_sum(
        npx.sum(
            vol_t
            * settings.grav
            / settings.rho_0
            * (
                -vs.int_drhodT[2:-2, 2:-2, :, vs.tau] * vs.dtemp_hmix[2:-2, 2:-2, :]
                - vs.int_drhodS[2:-2, 2:-2, :, vs.tau] * vs.dsalt_hmix[2:-2, 2:-2, :]
            )
        )
    )

    dP_vmix = global_sum(
        npx.sum(
            vol_t
            * settings.grav
            / settings.rho_0
            * (
                -vs.int_drhodT[2:-2, 2:-2, :, vs.tau] * vs.dtemp_vmix[2:-2, 2:-2, :]
                - vs.int_drhodS[2:-2, 2:-2, :, vs.tau] * vs.dsalt_vmix[2:-2, 2:-2, :]
            )
        )
    )

    dP_m = global_sum(
        npx.sum(
            vol_t
            * settings.grav
            / settings.rho_0
            * (
                -vs.int_drhodT[2:-2, 2:-2, :, vs.tau] * vs.dtemp[2:-2, 2:-2, :, vs.tau]
                - vs.int_drhodS[2:-2, 2:-2, :, vs.tau] * vs.dsalt[2:-2, 2:-2, :, vs.tau]
            )
        )
    )

    dP_m_all = dP_m + dP_vmix + dP_hmix + dP_iso

    # changes of kinetic energy
    vol_u = vs.area_u[2:-2, 2:-2, npx.newaxis] * vs.dzt[npx.newaxis, npx.newaxis, :]
    vol_v = vs.area_v[2:-2, 2:-2, npx.newaxis] * vs.dzt[npx.newaxis, npx.newaxis, :]
    k_m = global_sum(
        npx.sum(
            vol_t
            * 0.5
            * (
                0.5 * (vs.u[2:-2, 2:-2, :, vs.tau] ** 2 + vs.u[1:-3, 2:-2, :, vs.tau] ** 2)
                + 0.5 * (vs.v[2:-2, 2:-2, :, vs.tau] ** 2)
                + vs.v[2:-2, 1:-3, :, vs.tau] ** 2
            )
        )
    )
    p_m = global_sum(npx.sum(vol_t * vs.Hd[2:-2, 2:-2, :, vs.tau]))
    dk_m = global_sum(
        npx.sum(
            vs.u[2:-2, 2:-2, :, vs.tau] * vs.du[2:-2, 2:-2, :, vs.tau] * vol_u
            + vs.v[2:-2, 2:-2, :, vs.tau] * vs.dv[2:-2, 2:-2, :, vs.tau] * vol_v
            + vs.u[2:-2, 2:-2, :, vs.tau] * vs.du_mix[2:-2, 2:-2, :] * vol_u
            + vs.v[2:-2, 2:-2, :, vs.tau] * vs.dv_mix[2:-2, 2:-2, :] * vol_v
        )
    )

    # K*Nsqr and KE and dyn. enthalpy dissipation
    vol_w = vs.area_t[2:-2, 2:-2, npx.newaxis] * vs.dzw[npx.newaxis, npx.newaxis, :] * vs.maskW[2:-2, 2:-2, :]
    vol_w = update_multiply(vol_w, at[:, :, -1], 0.5)

    def mean_w(var):
        return global_sum(npx.sum(var[2:-2, 2:-2, :] * vol_w))

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

    wrhom = global_sum(
        npx.sum(
            -vs.area_t[2:-2, 2:-2, npx.newaxis]
            * vs.maskW[2:-2, 2:-2, :-1]
            * (vs.p_hydro[2:-2, 2:-2, 1:] - vs.p_hydro[2:-2, 2:-2, :-1])
            * vs.w[2:-2, 2:-2, :-1, vs.tau]
        )
    )

    # wind work
    if runtime_settings.pyom_compatibility_mode:
        wind = global_sum(
            npx.sum(
                vs.u[2:-2, 2:-2, -1, vs.tau]
                * vs.surface_taux[2:-2, 2:-2]
                * vs.maskU[2:-2, 2:-2, -1]
                * vs.area_u[2:-2, 2:-2]
                + vs.v[2:-2, 2:-2, -1, vs.tau]
                * vs.surface_tauy[2:-2, 2:-2]
                * vs.maskV[2:-2, 2:-2, -1]
                * vs.area_v[2:-2, 2:-2]
            )
        )
    else:
        wind = global_sum(
            npx.sum(
                vs.u[2:-2, 2:-2, -1, vs.tau]
                * vs.surface_taux[2:-2, 2:-2]
                / settings.rho_0
                * vs.maskU[2:-2, 2:-2, -1]
                * vs.area_u[2:-2, 2:-2]
                + vs.v[2:-2, 2:-2, -1, vs.tau]
                * vs.surface_tauy[2:-2, 2:-2]
                / settings.rho_0
                * vs.maskV[2:-2, 2:-2, -1]
                * vs.area_v[2:-2, 2:-2]
            )
        )

    # meso-scale energy
    if settings.enable_eke:
        eke_m = mean_w(vs.eke[..., vs.tau])
        deke_m = global_sum(
            npx.sum(vol_w * (vs.eke[2:-2, 2:-2, :, vs.taup1] - vs.eke[2:-2, 2:-2, :, vs.tau]) / settings.dt_tracer)
        )
        eke_diss = mean_w(vs.eke_diss_iw)
        eke_diss_tke = mean_w(vs.eke_diss_tke)
    else:
        eke_m = deke_m = eke_diss_tke = 0.0
        eke_diss = mdiss_gm + mdiss_h + mdiss_skew
        if not settings.enable_store_cabbeling_heat:
            eke_diss += -mdiss_hmix - mdiss_iso

    # small-scale energy
    if settings.enable_tke:
        dt_tke = settings.dt_mom
        tke_m = mean_w(vs.tke[..., vs.tau])
        dtke_m = mean_w((vs.tke[..., vs.taup1] - vs.tke[..., vs.tau]) / dt_tke)
        tke_diss = mean_w(vs.tke_diss)
        tke_forc = global_sum(
            npx.sum(
                vs.area_t[2:-2, 2:-2]
                * vs.maskW[2:-2, 2:-2, -1]
                * (vs.forc_tke_surface[2:-2, 2:-2] + vs.tke_surf_corr[2:-2, 2:-2])
            )
        )
    else:
        tke_m = dtke_m = tke_diss = tke_forc = 0.0

    # internal wave energy
    if settings.enable_idemix:
        iw_m = mean_w(vs.E_iw[..., vs.tau])
        diw_m = global_sum(
            npx.sum(vol_w * (vs.E_iw[2:-2, 2:-2, :, vs.taup1] - vs.E_iw[2:-2, 2:-2, :, vs.tau]) / vs.dt_tracer)
        )
        iw_diss = mean_w(vs.iw_diss)

        k = npx.maximum(1, vs.kbot[2:-2, 2:-2]) - 1
        mask = k[:, :, npx.newaxis] == npx.arange(settings.nz)[npx.newaxis, npx.newaxis, :]
        iwforc = global_sum(
            npx.sum(
                vs.area_t[2:-2, 2:-2]
                * (
                    vs.forc_iw_surface[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2, -1]
                    + npx.sum(mask * vs.forc_iw_bottom[2:-2, 2:-2, npx.newaxis] * vs.maskW[2:-2, 2:-2, :], axis=2)
                )
            )
        )
    else:
        iw_m = diw_m = iwforc = 0.0
        iw_diss = eke_diss

    if settings.enable_store_bottom_friction_tke:
        ke_tke_m = mdiss_v + mdiss_bot
        ke_iw_m = 0.0
    else:
        ke_tke_m = mdiss_v
        ke_iw_m = mdiss_bot

    hd_eke_m = -mdiss_skew
    tke_hd_m = -mdiss_vmix - mdiss_adv

    if not settings.enable_store_cabbeling_heat:
        hd_eke_m = hd_eke_m - mdiss_hmix - mdiss_iso
        tke_hd_m = tke_hd_m - mdiss_nonlin

    return KernelOutput(
        k_m=k_m,
        Hd_m=p_m,
        eke_m=eke_m,
        iw_m=iw_m,
        tke_m=tke_m,
        dk_m=dk_m,
        dHd_m=dP_m_all + mdiss_sources,
        deke_m=deke_m,
        diw_m=diw_m,
        dtke_m=dtke_m,
        dE_tot_m=dk_m + dP_m_all + mdiss_sources + deke_m + diw_m + dtke_m,
        wind_m=wind,
        dHd_sources_m=mdiss_sources,
        iw_forc_m=iwforc,
        tke_forc_m=tke_forc,
        ke_diss_m=mdiss_h + mdiss_v + mdiss_gm + mdiss_bot,
        Hd_diss_m=mdiss_vmix + mdiss_nonlin + mdiss_hmix + mdiss_adv + mdiss_iso + mdiss_skew,
        eke_diss_m=eke_diss + eke_diss_tke,
        iw_diss_m=iw_diss,
        tke_diss_m=tke_diss,
        adv_diss_m=mdiss_adv,
        ke_hd_m=wrhom,
        ke_eke_m=mdiss_h + mdiss_gm,
        hd_eke_m=-mdiss_skew,
        ke_tke_m=ke_tke_m,
        ke_iw_m=ke_iw_m,
        tke_hd_m=tke_hd_m,
        eke_tke_m=eke_diss_tke,
        eke_iw_m=eke_diss,
        cabb_m=mdiss_nonlin,
        cabb_iso_m=mdiss_hmix + mdiss_iso,
    )
