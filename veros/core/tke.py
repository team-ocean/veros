import math

from .. import veros_method
from . import cyclic, advection, utilities, numerics

@veros_method
def set_tke_diffusivities(veros):
    """
    set vertical diffusivities based on TKE model
    """
    Rinumber = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

    if veros.enable_tke:
        veros.sqrttke = np.sqrt(np.maximum(0., veros.tke[:,:,:,veros.tau]))
        """
        calculate buoyancy length scale
        """
        veros.mxl[...] = math.sqrt(2) * veros.sqrttke / np.sqrt(np.maximum(1e-12, veros.Nsqr[:,:,:,veros.tau])) * veros.maskW

        """
        apply limits for mixing length
        """
        if veros.tke_mxl_choice == 1:
            """
            bounded by the distance to surface/bottom
            """
            veros.mxl[...] = np.minimum(
                                np.minimum(veros.mxl, -veros.zw[np.newaxis, np.newaxis, :] \
                                                    + veros.dzw[np.newaxis, np.newaxis, :] * 0.5
                                          )
                                      , veros.ht[:, :, np.newaxis] + veros.zw[np.newaxis, np.newaxis, :]
                                      )
            veros.mxl[...] = np.maximum(veros.mxl, veros.mxl_min)
        elif veros.tke_mxl_choice == 2:
            """
            bound length scale as in mitgcm/OPA code

            Note that the following code doesn't vectorize. If critical for performance,
            consider re-implementing it in Cython.
            """
            if veros.backend_name == "bohrium":
                mxl = veros.mxl.copy2numpy()
                dzt = veros.dzt.copy2numpy()
            else:
                mxl = veros.mxl
                dzt = veros.dzt
            for k in xrange(veros.nz-2,-1,-1):
                mxl[:,:,k] = np.minimum(mxl[:,:,k], mxl[:,:,k+1] + dzt[k+1])
            mxl[:,:,-1] = np.minimum(mxl[:,:,-1], veros.mxl_min + dzt[-1])
            for k in xrange(1,veros.nz):
                mxl[:,:,k] = np.minimum(mxl[:,:,k], mxl[:,:,k-1] + dzt[k])
            veros.mxl[...] = np.maximum(np.asarray(mxl), veros.mxl_min)
        else:
            raise ValueError("unknown mixing length choice in tke_mxl_choice")

        """
        calculate viscosity and diffusivity based on Prandtl number
        """
        if veros.enable_cyclic_x:
            cyclic.setcyclic_x(veros.K_diss_v)
        veros.kappaM = np.minimum(veros.kappaM_max, veros.c_k * veros.mxl * veros.sqrttke)
        Rinumber = veros.Nsqr[:,:,:,veros.tau] / np.maximum(veros.K_diss_v / np.maximum(1e-12, veros.kappaM), 1e-12)
        if veros.enable_idemix:
            Rinumber = np.minimum(Rinumber, veros.kappaM * veros.Nsqr[:,:,:,veros.tau] \
                     / np.maximum(1e-12, veros.alpha_c * veros.E_iw[:,:,:,veros.tau]**2))
        veros.Prandtlnumber = np.maximum(1., np.minimum(10, 6.6 * Rinumber))
        veros.kappaH = veros.kappaM / veros.Prandtlnumber
        veros.kappaM = np.maximum(veros.kappaM_min, veros.kappaM)
    else:
        veros.kappaM[...] = veros.kappaM_0
        veros.kappaH[...] = veros.kappaH_0
        if veros.enable_hydrostatic:
            """
            simple convective adjustment
            """
            veros.kappaH[...] = np.where(veros.Nsqr[:,:,:,veros.tau] < 0.0, 1.0, veros.kappaH)

@veros_method
def integrate_tke(veros):
    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    veros.dt_tke = veros.dt_mom  # use momentum time step to prevent spurious oscillations

    """
    Sources and sinks by vertical friction, vertical mixing, and non-conservative advection
    """
    forc = veros.K_diss_v - veros.P_diss_v - veros.P_diss_adv

    """
    store transfer due to vertical mixing from dyn. enthalpy by non-linear eq.of
    state either to TKE or to heat
    """
    if not veros.enable_store_cabbeling_heat:
        forc[...] += -veros.P_diss_nonlin

    """
    transfer part of dissipation of EKE to TKE
    """
    if veros.enable_eke:
        forc[...] += veros.eke_diss_tke

    if veros.enable_idemix:
        """
        transfer dissipation of internal waves to TKE
        """
        forc[...] += veros.iw_diss
        """
        store bottom friction either in TKE or internal waves
        """
        if veros.enable_store_bottom_friction_tke:
            forc[...] += veros.K_diss_bot
    else: # short-cut without idemix
        if veros.enable_eke:
            forc[...] += veros.eke_diss_iw
        else: # and without EKE model
            if veros.enable_store_cabbeling_heat:
                forc[...] += veros.K_diss_gm + veros.K_diss_h - veros.P_diss_skew \
                        - veros.P_diss_hmix  - veros.P_diss_iso
            else:
                forc[...] += veros.K_diss_gm + veros.K_diss_h - veros.P_diss_skew
        forc[...] += veros.K_diss_bot

    """
    vertical mixing and dissipation of TKE
    """
    ks = veros.kbot[2:-2, 2:-2] - 1

    a_tri = np.zeros((veros.nx,veros.ny,veros.nz))
    b_tri = np.zeros((veros.nx,veros.ny,veros.nz))
    c_tri = np.zeros((veros.nx,veros.ny,veros.nz))
    d_tri = np.zeros((veros.nx,veros.ny,veros.nz))
    delta = np.zeros((veros.nx,veros.ny,veros.nz))

    delta[:,:,:-1] = veros.dt_tke / veros.dzt[np.newaxis, np.newaxis, 1:] * veros.alpha_tke * 0.5 \
                    * (veros.kappaM[2:-2, 2:-2, :-1] + veros.kappaM[2:-2, 2:-2, 1:])

    a_tri[:,:,1:-1] = -delta[:,:,:-2] / veros.dzw[np.newaxis,np.newaxis,1:-1]
    a_tri[:,:,-1] = -delta[:,:,-2] / (0.5 * veros.dzw[-1])

    b_tri[:,:,1:-1] = 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / veros.dzw[np.newaxis, np.newaxis, 1:-1] \
                        + veros.dt_tke * veros.c_eps * veros.sqrttke[2:-2, 2:-2, 1:-1] / veros.mxl[2:-2, 2:-2, 1:-1]
    b_tri[:,:,-1] = 1 + delta[:,:,-2] / (0.5 * veros.dzw[-1]) \
                        + veros.dt_tke * veros.c_eps / veros.mxl[2:-2, 2:-2, -1] * veros.sqrttke[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / veros.dzw[np.newaxis,np.newaxis,:] \
                        + veros.dt_tke * veros.c_eps / veros.mxl[2:-2, 2:-2, :] * veros.sqrttke[2:-2, 2:-2, :]

    c_tri[:,:,:-1] = -delta[:,:,:-1] / veros.dzw[np.newaxis,np.newaxis,:-1]

    d_tri[...] = veros.tke[2:-2, 2:-2, :, veros.tau] + veros.dt_tke * forc[2:-2, 2:-2, :]
    d_tri[:,:,-1] += veros.dt_tke * veros.forc_tke_surface[2:-2, 2:-2] / (0.5 * veros.dzw[-1])

    sol, water_mask = utilities.solve_implicit(veros, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    veros.tke[2:-2, 2:-2, :, veros.taup1] = np.where(water_mask, sol, veros.tke[2:-2, 2:-2, :, veros.taup1])

    """
    store tke dissipation for diagnostics
    """
    veros.tke_diss[...] = veros.c_eps / veros.mxl * veros.sqrttke * veros.tke[:,:,:,veros.taup1]

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    mask = veros.tke[2:-2, 2:-2, -1, veros.taup1] < 0.0
    veros.tke_surf_corr[...] = 0.
    veros.tke_surf_corr[2:-2, 2:-2] = np.where(mask, -veros.tke[2:-2, 2:-2, -1, veros.taup1] * 0.5 * veros.dzw[-1] / veros.dt_tke, 0.)
    veros.tke[2:-2, 2:-2, -1, veros.taup1] = np.maximum(0., veros.tke[2:-2, 2:-2, -1, veros.taup1])

    if veros.enable_tke_hor_diffusion:
        """
        add tendency due to lateral diffusion
        """
        veros.flux_east[:-1, :, :] = veros.K_h_tke * (veros.tke[1:, :, :, veros.tau] - veros.tke[:-1, :, :, veros.tau]) \
                                    / (veros.cost[np.newaxis, :, np.newaxis] * veros.dxu[:-1, np.newaxis, np.newaxis]) * veros.maskU[:-1, :, :]
        if veros.pyom_compatiblity_mode:
            veros.flux_east[-5,:,:] = 0.
        else:
            veros.flux_east[-1,:,:] = 0.
        veros.flux_north[:, :-1, :] = veros.K_h_tke * (veros.tke[:, 1:, :, veros.tau] - veros.tke[:, :-1, :, veros.tau]) \
                                     / veros.dyu[np.newaxis, :-1, np.newaxis] * veros.maskV[:, :-1, :] * veros.cosu[np.newaxis, :-1, np.newaxis]
        veros.flux_north[:,-1,:] = 0.
        veros.tke[2:-2, 2:-2, :, veros.taup1] += veros.dt_tke * veros.maskW[2:-2, 2:-2, :] * \
                                ((veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :]) \
                                   / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                                + (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :]) \
                                   / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    if veros.enable_tke_superbee_advection:
        advection.adv_flux_superbee_wgrid(veros,veros.flux_east,veros.flux_north,veros.flux_top,veros.tke[:,:,:,veros.tau])
    if veros.enable_tke_upwind_advection:
        advection.adv_flux_upwind_wgrid(veros,veros.flux_east,veros.flux_north,veros.flux_top,veros.tke[:,:,:,veros.tau])
    if veros.enable_tke_superbee_advection or veros.enable_tke_upwind_advection:
        veros.dtke[2:-2, 2:-2, :, veros.tau] = veros.maskW[2:-2, 2:-2, :] * (-(veros.flux_east[2:-2, 2:-2, :] - veros.flux_east[1:-3, 2:-2, :]) \
                                                                           / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                                                                         - (veros.flux_north[2:-2, 2:-2, :] - veros.flux_north[2:-2, 1:-3, :]) \
                                                                           / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dyt[np.newaxis, 2:-2, np.newaxis]))
        veros.dtke[:,:,0,veros.tau] += -veros.flux_top[:,:,0] / veros.dzw[0]
        veros.dtke[:,:,1:-1,veros.tau] += -(veros.flux_top[:,:,1:-1] - veros.flux_top[:,:,:-2]) / veros.dzw[1:-1]
        veros.dtke[:,:,-1,veros.tau] += -(veros.flux_top[:,:,-1] - veros.flux_top[:,:,-2]) / (0.5 * veros.dzw[-1])
        """
        Adam Bashforth time stepping
        """
        veros.tke[:,:,:,veros.taup1] += veros.dt_tracer * ((1.5 + veros.AB_eps) * veros.dtke[:,:,:,veros.tau] \
                                        - (0.5 + veros.AB_eps) * veros.dtke[:,:,:,veros.taum1])
