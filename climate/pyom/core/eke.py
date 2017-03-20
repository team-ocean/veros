import math

from .. import pyom_method
from . import utilities, advection

@pyom_method
def init_eke(pyom):
    """
    Initialize EKE
    """
    if pyom.enable_eke_leewave_dissipation:
        pyom.hrms_k0[...] = np.maximum(pyom.eke_hrms_k0_min, 2 / pyom.pi * pyom.eke_topo_hrms**2 / np.maximum(1e-12, pyom.eke_topo_lam)**1.5)

@pyom_method
def set_eke_diffusivities(pyom):
    """
    set skew diffusivity K_gm and isopycnal diffusivity K_iso
    set also vertical viscosity if TEM formalism is chosen
    """
    C_rossby = np.zeros((pyom.nx+4, pyom.ny+4))

    if pyom.enable_eke:
        """
        calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
        """
        C_rossby[...] = np.sum(np.sqrt(np.maximum(0.,pyom.Nsqr[:,:,:,pyom.tau])) * pyom.dzw[None, None, :] * pyom.maskW[:,:,:] / pyom.pi, axis=2)
        pyom.L_rossby[...] = np.minimum(C_rossby / np.maximum(np.abs(pyom.coriolis_t), 1e-16), \
                                          np.sqrt(C_rossby / np.maximum(2 * pyom.beta, 1e-16)))
        """
        calculate vertical viscosity and skew diffusivity
        """
        pyom.sqrteke = np.sqrt(np.maximum(0.,pyom.eke[:,:,:,pyom.tau]))
        pyom.L_rhines[...] = np.sqrt(pyom.sqrteke / np.maximum(pyom.beta[...,None], 1e-16))
        pyom.eke_len[...] = np.maximum(pyom.eke_lmin, np.minimum(pyom.eke_cross * pyom.L_rossby[...,None], pyom.eke_crhin * pyom.L_rhines))
        pyom.K_gm[...] = np.minimum(pyom.eke_k_max, pyom.eke_c_k * pyom.eke_len * pyom.sqrteke)
    else:
        """
        use fixed GM diffusivity
        """
        pyom.K_gm[...] = pyom.K_gm_0

    if pyom.enable_TEM_friction:
        pyom.kappa_gm[...] = pyom.K_gm * np.minimum(0.01, pyom.coriolis_t[...,None]**2 \
                               / np.maximum(1e-9, pyom.Nsqr[...,pyom.tau])) * pyom.maskW
    if pyom.enable_eke and pyom.enable_eke_isopycnal_diffusion:
        pyom.K_iso[...] = pyom.K_gm
    else:
        pyom.K_iso[...] = pyom.K_iso_0 # always constant

@pyom_method
def integrate_eke(pyom):
    """
    integrate EKE equation on W grid
    """
    c_int = np.zeros((pyom.nx+4, pyom.ny+4, pyom.nz))


    """
    forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
    """
    forc = pyom.K_diss_h + pyom.K_diss_gm  - pyom.P_diss_skew

    """
    store transfer due to isopycnal and horizontal mixing from dyn. enthalpy
    by non-linear eq.of state either to EKE or to heat
    """
    if not pyom.enable_store_cabbeling_heat:
        forc[...] += -pyom.P_diss_hmix - pyom.P_diss_iso

    """
    coefficient for dissipation of EKE:
    by lee wave generation, Ri-dependent interior loss of balance and bottom friction
    """
    if pyom.enable_eke_leewave_dissipation:
        """
        by lee wave generation
        """
        pyom.c_lee[...] = 0.
        ks = pyom.kbot[2:-2, 2:-2] - 1
        ki = np.arange(pyom.nz)[np.newaxis, np.newaxis, :]
        boundary_mask = (ks >= 0) & (ks < pyom.nz-1)
        full_mask = boundary_mask[:,:,None] & (ki == ks[:,:,None])
        fxa = np.maximum(0, pyom.Nsqr[2:-2, 2:-2, :, pyom.tau])**0.25
        fxa *= 1.5 * fxa / np.sqrt(np.maximum(1e-6, np.abs(pyom.coriolis_t[2:-2, 2:-2, None]))) - 2
        pyom.c_lee[2:-2, 2:-2] += boundary_mask * np.sum((pyom.c_lee0 * pyom.hrms_k0[2:-2, 2:-2, None] * np.sqrt(pyom.sqrteke[2:-2, 2:-2, :]) \
                                        * np.maximum(0, fxa) / pyom.dzw[None, None, :]) * full_mask, axis=-1)

        """
        Ri-dependent dissipation by interior loss of balance
        """
        pyom.c_Ri_diss[...] = 0
        uz = (((pyom.u[1:,1:,1:,pyom.tau] - pyom.u[1:,1:,:-1,pyom.tau]) / pyom.dzt[None, None, :-1] * pyom.maskU[1:, 1:, :-1])**2 \
              + ((pyom.u[:-1, 1:, 1:, pyom.tau] - pyom.u[:-1, 1:, :-1, pyom.tau]) / pyom.dzt[None, None, :-1] * pyom.maskU[:-1, 1:, :-1])**2) \
              / (pyom.maskU[1:, 1:, :-1] + pyom.maskU[:-1, 1:, :-1] + 1e-18)
        vz = (((pyom.v[1:,1:,1:,pyom.tau] - pyom.v[1:,1:,:-1,pyom.tau]) / pyom.dzt[None, None, :-1] * pyom.maskV[1:,1:,:-1])**2 \
                    + ((pyom.v[1:,:-1,1:,pyom.tau] - pyom.v[1:,:-1,:-1,pyom.tau]) / pyom.dzt[None, None, :-1] * pyom.maskV[1:,:-1,:-1])**2) \
                    / (pyom.maskV[1:,1:,:-1] + pyom.maskV[1:,:-1,:-1] + 1e-18)
        Ri = np.maximum(1e-8, pyom.Nsqr[1:,1:,:-1,pyom.tau]) / (uz + vz + 1e-18)
        fxa = 1 - 0.5 * (1. + np.tanh((Ri - pyom.eke_Ri0) / pyom.eke_Ri1))
        pyom.c_Ri_diss[1:,1:,:-1] = pyom.maskW[1:,1:,:-1] * fxa * pyom.eke_int_diss0
        pyom.c_Ri_diss[:,:,-1] = pyom.c_Ri_diss[:,:,-2] * pyom.maskW[:,:,-1]

        """
        vertically integrate Ri-dependent dissipation and EKE
        """
        a_loc = np.sum(pyom.c_Ri_diss[:,:,:-1] * pyom.eke[:,:,:-1,pyom.tau] * pyom.maskW[:,:,:-1] * pyom.dzw[:-1], axis=2)
        b_loc = np.sum(pyom.eke[:,:,:-1,pyom.tau] * pyom.maskW[:,:,:-1] * pyom.dzw[:-1], axis=2)
        a_loc += pyom.c_Ri_diss[:,:,-1] * pyom.eke[:,:,-1,pyom.tau] * pyom.maskW[:,:,-1] * pyom.dzw[-1] * 0.5
        b_loc += pyom.eke[:,:,-1,pyom.tau] * pyom.maskW[:,:,-1] * pyom.dzw[-1] * 0.5

        """
        add bottom fluxes by lee waves and bottom friction to a_loc
        """
        a_loc[2:-2, 2:-2] += np.sum((pyom.c_lee[2:-2,2:-2,None] * pyom.eke[2:-2,2:-2,:,pyom.tau] \
                                * pyom.maskW[2:-2,2:-2,:] * pyom.dzw[None, None, :] \
                           + 2 * pyom.eke_r_bot * pyom.eke[2:-2,2:-2,:,pyom.tau] * math.sqrt(2.0) * pyom.sqrteke[2:-2,2:-2,:] \
                                * pyom.maskW[2:-2,2:-2,:]) * full_mask, axis=-1) * boundary_mask

        """
        dissipation constant is vertically integrated forcing divided by
        vertically integrated EKE to account for vertical EKE radiation
        """
        mask = b_loc > 0
        a_loc[...] = np.where(mask, a_loc/(b_loc+1e-20), 0.)
        c_int[...] = a_loc[:,:,None]
    else:
        """
        dissipation by local interior loss of balance with constant coefficient
        """
        c_int[...] = pyom.eke_c_eps * pyom.sqrteke / pyom.eke_len * pyom.maskW

    """
    vertical diffusion of EKE,forcing and dissipation
    """
    ks = pyom.kbot[2:-2, 2:-2] - 1
    delta, a_tri, b_tri, c_tri, d_tri = (np.zeros((pyom.nx, pyom.ny, pyom.nz)) for _ in range(5))
    delta[:,:,:-1] = pyom.dt_tracer / pyom.dzt[None, None, 1:] * 0.5 \
                 * (pyom.kappaM[2:-2, 2:-2, :-1] + pyom.kappaM[2:-2, 2:-2, 1:]) * pyom.alpha_eke
    a_tri[:, :, 1:-1] = -delta[:,:,:-2] / pyom.dzw[1:-1]
    a_tri[:, :, -1] = -delta[:,:,-2] / (0.5 * pyom.dzw[-1])
    b_tri[:, :, 1:-1] = 1 + (delta[:,:,1:-1] + delta[:,:,:-2]) / pyom.dzw[1:-1] + pyom.dt_tracer * c_int[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:,:,-2] / (0.5 * pyom.dzw[-1]) + pyom.dt_tracer * c_int[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / pyom.dzw[None, None, :] + pyom.dt_tracer * c_int[2:-2, 2:-2, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / pyom.dzw[None, None, :-1]
    d_tri[:, :, :] = pyom.eke[2:-2, 2:-2, :, pyom.tau] + pyom.dt_tracer * forc[2:-2, 2:-2, :]
    sol, water_mask = utilities.solve_implicit(pyom, ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    pyom.eke[2:-2, 2:-2, :, pyom.taup1] = np.where(water_mask, sol, pyom.eke[2:-2, 2:-2, :, pyom.taup1])

    """
    store eke dissipation
    """
    if pyom.enable_eke_leewave_dissipation:
        pyom.eke_diss_iw[...] = 0.
        pyom.eke_diss_tke[...] = pyom.c_Ri_diss * pyom.eke[:,:,:,pyom.taup1]

        """
        flux by lee wave generation and bottom friction
        """
        pyom.eke_diss_iw[2:-2, 2:-2, :] += (pyom.c_lee[2:-2, 2:-2, None] * pyom.eke[2:-2, 2:-2, :, pyom.taup1] \
                                                       * pyom.maskW[2:-2, 2:-2, :]) * full_mask
        pyom.eke_diss_tke[2:-2, 2:-2, :] += (2 * pyom.eke_r_bot * pyom.eke[2:-2, 2:-2, :, pyom.taup1] * math.sqrt(2.0) \
                                    * pyom.sqrteke[2:-2, 2:-2, :] * pyom.maskW[2:-2, 2:-2, :] / pyom.dzw[None, None, :]) * full_mask

        """
        account for sligthly incorrect integral of dissipation due to time stepping
        """
        a_loc = np.sum((pyom.eke_diss_iw[:,:,:-1] + pyom.eke_diss_tke[:,:,:-1]) * pyom.dzw[None, None, :-1], axis=2)
        b_loc = np.sum(c_int[:,:,:-1] * pyom.eke[:,:,:-1,pyom.taup1] * pyom.dzw[None, None, :-1], axis=2)
        a_loc += (pyom.eke_diss_iw[:,:,-1] + pyom.eke_diss_tke[:,:,-1]) * pyom.dzw[-1] * 0.5
        b_loc += c_int[:,:,-1] * pyom.eke[:,:,-1,pyom.taup1] * pyom.dzw[-1] * 0.5
        mask = a_loc != 0.
        b_loc[...] = np.where(mask, b_loc / (a_loc+1e-20), 0.)
        pyom.eke_diss_iw[...] *= b_loc[:,:,None]
        pyom.eke_diss_tke[...] *= b_loc[:,:,None]

        """
        store diagnosed flux by lee waves and bottom friction
        """
        pyom.eke_lee_flux[2:-2, 2:-2] = np.where(boundary_mask, np.sum(pyom.c_lee[2:-2, 2:-2, None] * pyom.eke[2:-2, 2:-2, :, pyom.taup1] \
                                                        * pyom.dzw[None, None, :] * full_mask, axis=-1)
                                                , pyom.eke_lee_flux[2:-2, 2:-2])
        pyom.eke_bot_flux[2:-2, 2:-2] = np.where(boundary_mask, np.sum(2 * pyom.eke_r_bot * pyom.eke[2:-2, 2:-2, :, pyom.taup1] \
                                                        * math.sqrt(2.0) * pyom.sqrteke[2:-2, 2:-2, :] * full_mask, axis=-1)
                                                , pyom.eke_bot_flux[2:-2, 2:-2])
    else:
        pyom.eke_diss_iw = c_int * pyom.eke[:,:,:,pyom.taup1]
        pyom.eke_diss_tke[...] = 0.

    """
    add tendency due to lateral diffusion
    """
    pyom.flux_east[:-1,:,:] = 0.5 * np.maximum(500., pyom.K_gm[:-1,:,:] + pyom.K_gm[1:,:,:]) \
                            * (pyom.eke[1:,:,:,pyom.tau] - pyom.eke[:-1,:,:,pyom.tau]) \
                            / (pyom.cost[None,:,None] * pyom.dxu[:-1, None, None]) * pyom.maskU[:-1,:,:]
    pyom.flux_east[-1,:,:] = 0.
    pyom.flux_north[:,:-1,:] = 0.5 * np.maximum(500., pyom.K_gm[:,:-1,:] + pyom.K_gm[:,1:,:]) \
                             * (pyom.eke[:,1:,:,pyom.tau] - pyom.eke[:,:-1,:,pyom.tau]) \
                             / pyom.dyu[None,:-1,None] * pyom.maskV[:,:-1,:] * pyom.cosu[None,:-1,None]
    pyom.flux_north[:,-1,:] = 0.
    pyom.eke[2:-2,2:-2,:,pyom.taup1] += pyom.dt_tracer * pyom.maskW[2:-2,2:-2,:] \
                                 * ((pyom.flux_east[2:-2,2:-2,:] - pyom.flux_east[1:-3,2:-2,:]) \
                                 / (pyom.cost[None,2:-2,None] * pyom.dxt[2:-2,None,None]) \
                                 + (pyom.flux_north[2:-2,2:-2,:] - pyom.flux_north[2:-2,1:-3,:]) \
                                 / (pyom.cost[None,2:-2,None] * pyom.dyt[None,2:-2,None]))

    """
    add tendency due to advection
    """
    if pyom.enable_eke_superbee_advection:
        advection.adv_flux_superbee_wgrid(pyom, pyom.flux_east, pyom.flux_north, pyom.flux_top, pyom.eke[:,:,:,pyom.tau])
    if pyom.enable_eke_upwind_advection:
        advection.adv_flux_upwind_wgrid(pyom, pyom.flux_east, pyom.flux_north, pyom.flux_top, pyom.eke[:,:,:,pyom.tau])
    if pyom.enable_eke_superbee_advection or pyom.enable_eke_upwind_advection:
        pyom.deke[2:-2,2:-2,:,pyom.tau] = pyom.maskW[2:-2,2:-2,:] * (-(pyom.flux_east[2:-2,2:-2,:] - pyom.flux_east[1:-3,2:-2,:]) \
                                       / (pyom.cost[None,2:-2,None] * pyom.dxt[2:-2,None,None]) \
                                    - (pyom.flux_north[2:-2,2:-2,:] - pyom.flux_north[2:-2,1:-3,:]) \
                                       / (pyom.cost[None,2:-2,None] * pyom.dyt[None,2:-2,None]))
        pyom.deke[:,:,0,pyom.tau] += -pyom.flux_top[:,:,0] / pyom.dzw[0]
        pyom.deke[:,:,1:-1,pyom.tau] += -(pyom.flux_top[:,:,1:-1] - pyom.flux_top[:,:,:-2]) / pyom.dzw[None, None, 1:-1]
        pyom.deke[:,:,-1,pyom.tau] += -(pyom.flux_top[:,:,-1] - pyom.flux_top[:,:,-2]) / (0.5 * pyom.dzw[-1])
        """
        Adam Bashforth time stepping
        """
        pyom.eke[:,:,:,pyom.taup1] += pyom.dt_tracer * ((1.5 + pyom.AB_eps) * pyom.deke[:,:,:,pyom.tau] \
                                                      - (0.5 + pyom.AB_eps) * pyom.deke[:,:,:,pyom.taum1])
