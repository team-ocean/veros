import numpy as np

from climate import make_slice
from climate.pyom.utilities import pad_z_edges

def _calc_cr(rjp,rj,rjm,vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    cr = np.zeros_like(rj)
    mask_rj = rj == 0.
    mask_vel = vel > 0
    mask1 = ~mask_rj & mask_vel
    cr[mask1] = rjm[mask1] / rj[mask1]
    mask2 = ~mask_rj & ~mask_vel
    cr[mask2] = rjp[mask2] / rj[mask2]
    mask3 = mask_rj & mask_vel
    cr[mask3] = rjm[mask3] * 1e20
    mask4 = mask_rj & ~mask_vel
    cr[mask4] = rjp[mask4] * 1e20
    return cr


def _adv_superbee(vel, var, mask, dx, axis, pyom):
    limiter = lambda cr: np.maximum(0.,np.maximum(np.minimum(1.,2*cr), np.minimum(2.,cr)))
    velfac = 1
    if axis == 0:
        i, ii = make_slice(1,-2)
        j, jj = make_slice(2,-2)
        k, kk = make_slice()
        sm1, s, sp1, sp2 = ((ii+n,j,k) for n in range(-1,3))
        dx = pyom.cost[None, j, None] * dx[i, None, None]
    elif axis == 1:
        i, ii = make_slice(2,-2)
        j, jj = make_slice(1,-2)
        k, kk = make_slice()
        sm1, s, sp1, sp2 = ((i,jj+n,k) for n in range(-1,3))
        dx = (pyom.cost * dx)[None,j,None]
        velfac = pyom.cosu[None, j, None]
    elif axis == 2:
        i, ii = make_slice(2,-2)
        j, jj = make_slice(2,-2)
        k, kk = make_slice(1,-2)
        vel, var, mask = (pad_z_edges(a) for a in (vel,var,mask))
        sm1, s, sp1, sp2 = ((i,j,kk+n) for n in range(-1,3))
        dx = dx[None,None,:-1]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    uCFL = np.abs(velfac * vel[s] * pyom.dt_tracer / dx)
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp,rj,rjm,vel[s]))
    return velfac * vel[s] * (var[sp1] + var[s]) * 0.5 - np.abs(velfac * vel[s]) * ((1.-cr) + uCFL*cr) * rj * 0.5


def adv_flux_2nd(adv_fe,adv_fn,adv_ft,var,pyom):
    """
    2th order advective tracer flux
    """
    i, ii = make_slice(1,-2)
    j, jj = make_slice(2,-2)
    k, kk = make_slice()
    adv_fe[i, j, k] = 0.5*(var[i,j,k] + var[ii+1,j,k]) * pyom.u[i,j,k,pyom.tau] * pyom.maskU[i,j,k]

    i, ii = make_slice(2,-2)
    j, jj = make_slice(1,-2)
    k, kk = make_slice()
    adv_fn[i,j,k] = pyom.cosu[None,j,None] * 0.5 * (var[i,j,k] + var[i,jj+1,k]) * pyom.v[i,j,k,pyom.tau] * pyom.maskV[i,j,k]

    i, ii = make_slice(2,-2)
    j, jj = make_slice(2,-2)
    k, kk = make_slice(None,-1)
    adv_ft[i,j,k] = 0.5 * (var[i,j,k] + var[i,j,kk+1]) * pyom.w[i,j,k,pyom.tau] * pyom.maskW[i,j,k]
    adv_ft[:,:,-1] = 0.


def adv_flux_superbee(adv_fe,adv_fn,adv_ft,var,pyom):
    """
    from MITgcm
    Calculates advection of a tracer
    using second-order interpolation with a flux limiter:
    \begin{equation*}
    F^x_{adv} = U \overline{ \theta }^i
    - \frac{1}{2} \left([ 1 - \psi(C_r) ] |U|
       + U \frac{u \Delta t}{\Delta x_c} \psi(C_r)
                 \right) \delta_i \theta
    \end{equation*}
    where the $\psi(C_r)$ is the limiter function and $C_r$ is
    the slope ratio.
    """
    adv_fe[1:-2, 2:-2, :] = _adv_superbee(pyom.u[..., pyom.tau], var, pyom.maskU, pyom.dxt, 0, pyom)
    adv_fn[2:-2, 1:-2, :] = _adv_superbee(pyom.v[..., pyom.tau], var, pyom.maskV, pyom.dyt, 1, pyom)
    adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(pyom.w[..., pyom.tau], var, pyom.maskW, pyom.dzt, 2, pyom)
    adv_ft[..., -1] = 0.


def calculate_velocity_on_wgrid(pyom):
    """
    calculates advection velocity for tracer on W grid
    """
    # lateral advection velocities on W grid
    pyom.u_wgrid[:,:,:-1] = pyom.u[:,:,1:,pyom.tau] * pyom.maskU[:,:,1:] * 0.5 * pyom.dzt[None,None,1:] / pyom.dzw[None,None,:-1] \
                          + pyom.u[:,:,:-1,pyom.tau] * pyom.maskU[:,:,:-1] * 0.5 * pyom.dzt[None,None,:-1] / pyom.dzw[None,None,:-1]
    pyom.v_wgrid[:,:,:-1] = pyom.v[:,:,1:,pyom.tau] * pyom.maskV[:,:,1:] * 0.5 * pyom.dzt[None,None,1:] / pyom.dzw[None,None,:-1] \
                          + pyom.v[:,:,:-1,pyom.tau] * pyom.maskV[:,:,:-1] * 0.5 * pyom.dzt[None,None,:-1] / pyom.dzw[None,None,:-1]
    pyom.u_wgrid[:,:,-1] = pyom.u[:,:,-1,pyom.tau] * pyom.maskU[:,:,-1] * 0.5 * pyom.dzt[-1] / pyom.dzw[-1]
    pyom.v_wgrid[:,:,-1] = pyom.v[:,:,-1,pyom.tau] * pyom.maskV[:,:,-1] * 0.5 * pyom.dzt[-1] / pyom.dzw[-1]

    # redirect velocity at bottom and at topography
    pyom.u_wgrid[:,:,0] = pyom.u_wgrid[:,:,0] + pyom.u[:,:,0,pyom.tau] * pyom.maskU[:,:,0] * 0.5 * pyom.dzt[0] / pyom.dzw[0]
    pyom.v_wgrid[:,:,0] = pyom.v_wgrid[:,:,0] + pyom.v[:,:,0,pyom.tau] * pyom.maskV[:,:,0] * 0.5 * pyom.dzt[0] / pyom.dzw[0]
    for k in xrange(pyom.nz-1): #TODO: vectorize
        mask = pyom.maskW[:-1, :, k] * pyom.maskW[1:, :, k] == 0
        pyom.u_wgrid[:-1, :, k+1][mask] += (pyom.u_wgrid[:-1, :, k] * pyom.dzw[None, None, k] / pyom.dzw[None, None, k+1])[mask]
        pyom.u_wgrid[:-1, :, k][mask] = 0.

        mask = pyom.maskW[:, :-1, k] * pyom.maskW[:, 1:, k] == 0
        pyom.v_wgrid[:, :-1, k+1][mask] += (pyom.v_wgrid[:, :-1, k] * pyom.dzw[None, None, k] / pyom.dzw[None, None, k+1])[mask]
        pyom.v_wgrid[:, :-1, k][mask] = 0.

    # vertical advection velocity on W grid from continuity
    pyom.w_wgrid[:, :, 0] = 0.
    pyom.w_wgrid[1:, 1:, :] = np.cumsum(-pyom.dzw[None, None, :] * \
                              ((pyom.u_wgrid[1:, 1:, :] - pyom.u_wgrid[:-1, 1:, :]) / (pyom.cost[None, 1:, None] * pyom.dxt[1:, None, None]) \
                               + (pyom.cosu[None, 1:, None] * pyom.v_wgrid[1:, 1:, :] - pyom.cosu[None, :-1, None] * pyom.v_wgrid[1:, :-1, :]) \
                                     / (pyom.cost[None, 1:, None] * pyom.dyt[None, 1:, None])), axis=2)



def adv_flux_superbee_wgrid(adv_fe,adv_fn,adv_ft,var,pyom):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = np.zeros_like(adv_fe)
    maskUtr[:-1, :, :] = pyom.maskW[1:, :, :] * pyom.maskW[:-1, :, :]
    adv_fe[1:-2, 2:-2, :] = _adv_superbee(pyom.u_wgrid, var, maskUtr, pyom.dxt, 0, pyom)

    maskVtr = np.zeros_like(adv_fn)
    maskVtr[:, :-1, :] = pyom.maskW[:, 1:, :] * pyom.maskW[:, :-1, :]
    adv_fn[2:-2, 1:-2, :] = _adv_superbee(pyom.v_wgrid, var, maskVtr, pyom.dyt, 1, pyom)

    maskWtr = np.zeros_like(adv_ft)
    maskWtr[:, :, :-1] = pyom.maskW[:, :, 1:] * pyom.maskW[:, :, :-1]
    adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(pyom.w_wgrid, var, maskWtr, pyom.dzw, 2, pyom)
    adv_ft[..., -1] = 0.0


def adv_flux_upwind_wgrid(adv_fe,adv_fn,adv_ft,var,pyom):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = pyom.maskW[2:-1, 2:-2, :] * pyom.maskW[1:-2, 2:-2, :]
    rj = (var[2:-1, 2:-2, :] - var[1:-2, 2:-2, :]) * maskUtr
    adv_fe[1:-2, 2:-2, :] = pyom.u_wgrid[1:-2, 2:-2, :] * (var[2:-1, 2:-2, :] + var[1:-2, 2:-2, :]) * 0.5 \
                            - np.abs(pyom.u_wgrid[1:-2, 2:-2, :]) * rj * 0.5

    maskVtr = pyom.maskW[2:-2, 2:-1, :] * pyom.maskW[2:-2, 1:-2, :]
    rj = (var[2:-2, 2:-1, :] - var[2:-2, 1:-2, :]) * maskVtr
    adv_fn[2:-2, 1:-2, :] = pyom.cosu[None, 1:-2, None] * pyom.v_wgrid[2:-2, 1:-2, :] * \
                            (var[2:-2, 2:-1, :] + var[2:-2, 1:-2, :]) * 0.5 \
                            - np.abs(pyom.cosu[None, 1:-2, None] * pyom.v_wgrid[2:-2, 1:-2, :]) * rj * 0.5

    maskWtr = pyom.maskW[2:-2, 2:-2, 1:] * pyom.maskW[2:-2, 2:-2, :-1]
    rj = (var[2:-2, 2:-2, 1:] - var[2:-2, 2:-2, :-1]) * maskWtr
    adv_ft[2:-2, 2:-2, :-1] = pyom.w_wgrid[2:-2, 2:-2, :-1] * (var[2:-2, 2:-2, 1:] + var[2:-2, 2:-2, :-1]) * 0.5 \
                              - np.abs(pyom.w_wgrid[2:-2, 2:-2, :-1]) * rj * 0.5
    adv_ft[:,:,-1] = 0.
