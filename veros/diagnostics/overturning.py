import os

from veros import logger

from veros.diagnostics.diagnostic import VerosDiagnostic
from veros.core import density
from veros.variables import Variable, allocate
from veros.distributed import global_sum
from veros.core.operators import numpy as np, update, update_add, at


SIGMA = Variable(
    'Sigma axis', ('sigma',), 'kg/m^3', 'Sigma axis', output=True,
    time_dependent=False, write_to_restart=True
)

VARIABLES = {
    'trans': Variable(
        'Meridional transport', ('yu', 'sigma'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'vsf_iso': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'vsf_depth': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'bolus_iso': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'bolus_depth': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
}


def zonal_sum(arr):
    return global_sum(np.sum(arr, axis=0), axis=0)


class Overturning(VerosDiagnostic):
    """Isopycnal overturning diagnostic. Computes and writes vertical streamfunctions
    (zonally averaged).
    """

    name = 'overturning'  #:
    output_path = '{identifier}.overturning.nc'  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.
    p_ref = 2000.  #: Reference pressure for isopycnals

    def __init__(self):
        self.sigma_var = SIGMA
        self.mean_variables = VARIABLES
        self.variables = self.mean_variables.copy()
        self.variables.update({'sigma': self.sigma_var})

    def initialize(self, state):
        vs = state.variables
        settings = state.settings

        self.nitts = 0
        self.nlevel = settings.nz * 4
        self._allocate(state)

        # sigma levels
        self.sige = density.get_potential_rho(state, 35., -2., press_ref=self.p_ref)
        self.sigs = density.get_potential_rho(state, 35., 30., press_ref=self.p_ref)
        self.dsig = float(self.sige - self.sigs) / (self.nlevel - 1)

        logger.debug(' Sigma ranges for overturning diagnostic:')
        logger.debug(' Start sigma0 = {:.1f}'.format(self.sigs))
        logger.debug(' End sigma0 = {:.1f}'.format(self.sige))
        logger.debug(' Delta sigma0 = {:.1e}'.format(self.dsig))
        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            logger.debug(' Also calculating overturning by eddy-driven velocities')

        self.sigma = self.sigs + self.dsig * np.arange(self.nlevel)

        # precalculate area below z levels
        self.zarea = update(self.zarea, at[2:-2, :], np.cumsum(zonal_sum(
            vs.dxt[2:-2, np.newaxis, np.newaxis]
            * vs.cosu[np.newaxis, 2:-2, np.newaxis]
            * vs.maskV[2:-2, 2:-2, :]) * vs.dzt[np.newaxis, :], axis=1))

        self.initialize_output(state, self.variables,
                               var_data={'sigma': self.sigma},
                               extra_dimensions={'sigma': self.nlevel})

    def _allocate(self, state):
        settings = state.settings

        self.sigma = allocate(state.dimensions, (self.nlevel,))
        self.zarea = allocate(state.dimensions, ('yu', 'zt'))
        self.trans = allocate(state.dimensions, ('yu', self.nlevel))
        self.vsf_iso = allocate(state.dimensions, ('yu', 'zt'))
        self.vsf_depth = allocate(state.dimensions, ('yu', 'zt'))

        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            self.bolus_iso = allocate(state.dimensions, ('yu', 'zt'))
            self.bolus_depth = allocate(state.dimensions, ('yu', 'zt'))

    def diagnose(self, state):
        vs = state.variables
        settings = state.settings

        # sigma at p_ref
        sig_loc = allocate(state.dimensions, ('xt', 'yt', 'zt'))
        sig_loc = update(sig_loc, at[2:-2, 2:-1, :], density.get_rho(state,
                                                 vs.salt[2:-2, 2:-1, :, vs.tau],
                                                 vs.temp[2:-2, 2:-1, :, vs.tau],
                                                 self.p_ref)
        )

        # transports below isopycnals and area below isopycnals
        sig_loc_face = 0.5 * (sig_loc[2:-2, 2:-2, :] + sig_loc[2:-2, 3:-1, :])
        trans = allocate(state.dimensions, ('yu', self.nlevel))
        z_sig = allocate(state.dimensions, ('yu', self.nlevel))

        fac = (vs.dxt[2:-2, np.newaxis, np.newaxis]
                * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                * vs.dzt[np.newaxis, np.newaxis, :]
                * vs.maskV[2:-2, 2:-2, :])

        for m in range(self.nlevel):
            # NOTE: vectorized version would be O(N^4) in memory
            # consider cythonizing if performance-critical
            mask = sig_loc_face > self.sigma[m]
            trans = update(trans, at[2:-2, m], zonal_sum(np.sum(vs.v[2:-2, 2:-2, :, vs.tau] * fac * mask, axis=2)))
            z_sig = update(z_sig, at[2:-2, m], zonal_sum(np.sum(fac * mask, axis=2)))

        self.trans = self.trans + trans

        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            bolus_trans = allocate(state.dimensions, ('yu', self.nlevel))

            # eddy-driven transports below isopycnals
            for m in range(self.nlevel):
                # NOTE: see above
                mask = sig_loc_face > self.sigma[m]
                bolus_trans = update(bolus_trans, at[2:-2, m], zonal_sum(
                    np.sum(
                        (vs.B1_gm[2:-2, 2:-2, 1:] - vs.B1_gm[2:-2, 2:-2, :-1])
                        * vs.dxt[2:-2, np.newaxis, np.newaxis]
                        * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                        * vs.maskV[2:-2, 2:-2, 1:]
                        * mask[:, :, 1:],
                        axis=2
                    )
                    +
                    vs.B1_gm[2:-2, 2:-2, 0]
                    * vs.dxt[2:-2, np.newaxis]
                    * vs.cosu[np.newaxis, 2:-2]
                    * vs.maskV[2:-2, 2:-2, 0]
                    * mask[:, :, 0]
                ))

        # streamfunction on geopotentials
        self.vsf_depth = update_add(self.vsf_depth, at[2:-2, :], np.cumsum(zonal_sum(
            vs.dxt[2:-2, np.newaxis, np.newaxis]
            * vs.cosu[np.newaxis, 2:-2, np.newaxis]
            * vs.v[2:-2, 2:-2, :, vs.tau]
            * vs.maskV[2:-2, 2:-2, :]) * vs.dzt[np.newaxis, :], axis=1))

        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            # streamfunction for eddy driven velocity on geopotentials
            self.bolus_depth = update_add(self.bolus_depth, at[2:-2, :], zonal_sum(
                vs.dxt[2:-2, np.newaxis, np.newaxis]
                * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                * vs.B1_gm[2:-2, 2:-2, :]))

        # interpolate from isopycnals to depth
        self.vsf_iso = update_add(self.vsf_iso, at[2:-2, :], self._interpolate_along_axis(
                                                              z_sig[2:-2, :], trans[2:-2, :],
                                                              self.zarea[2:-2, :], 1))
        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            self.bolus_iso = update_add(self.bolus_iso, at[2:-2, :], self._interpolate_along_axis(
                                                                    z_sig[2:-2, :], bolus_trans[2:-2, :],
                                                                    self.zarea[2:-2, :], 1))

        self.nitts += 1

    def _interpolate_along_axis(self, coords, arr, interp_coords, axis=0):
        # TODO: clean up this mess
        import numpy as onp

        if coords.ndim == 1:
            if len(coords) != arr.shape[axis]:
                raise ValueError('Coordinate shape must match array shape along axis')
        elif coords.ndim == arr.ndim:
            if coords.shape != arr.shape:
                raise ValueError('Coordinate shape must match array shape')
        else:
            raise ValueError('Coordinate shape must match array dimensions')

        if axis != 0:
            arr = onp.moveaxis(arr, axis, 0)
            coords = onp.moveaxis(coords, axis, 0)
            interp_coords = onp.moveaxis(interp_coords, axis, 0)

        diff = coords[onp.newaxis, :, ...] - interp_coords[:, onp.newaxis, ...]
        diff_m = onp.where(diff <= 0., onp.abs(diff), onp.inf)
        diff_p = onp.where(diff > 0., onp.abs(diff), onp.inf)
        i_m = onp.asarray(onp.argmin(diff_m, axis=1))
        i_p = onp.asarray(onp.argmin(diff_p, axis=1))
        mask = onp.all(onp.isinf(diff_m), axis=1)
        i_m[mask] = i_p[mask]
        mask = onp.all(onp.isinf(diff_p), axis=1)
        i_p[mask] = i_m[mask]
        full_shape = (slice(None),) + (onp.newaxis,) * (arr.ndim - 1)
        if coords.ndim == 1:
            i_p_full = i_p[full_shape] * onp.ones(arr.shape)
            i_m_full = i_m[full_shape] * onp.ones(arr.shape)
        else:
            i_p_full = i_p
            i_m_full = i_m
        ii = onp.indices(i_p_full.shape)
        i_p_slice = (i_p_full,) + tuple(ii[1:])
        i_m_slice = (i_m_full,) + tuple(ii[1:])
        dx = (coords[i_p_slice] - coords[i_m_slice])
        pos = onp.where(dx == 0., 0., (coords[i_p_slice] - interp_coords) / (dx + 1e-12))
        return np.asarray(onp.moveaxis(arr[i_p_slice] * (1. - pos) + arr[i_m_slice] * pos, 0, axis))

    def output(self, state):
        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state, self.variables,
                                   var_data={'sigma': self.sigma},
                                   extra_dimensions={'sigma': self.nlevel})

        if self.nitts > 0:
            for var in self.mean_variables.keys():
                setattr(self, var, getattr(self, var) / self.nitts)

        var_metadata = {key: var for key, var in self.mean_variables.items()
                        if var.output and var.time_dependent}
        var_data = {key: getattr(self, key) for key, var in self.mean_variables.items()
                    if var.output and var.time_dependent}
        self.write_output(state, var_metadata, var_data)

        self.nitts = 0
        for var in self.mean_variables.keys():
            setattr(self, var, np.zeros_like(getattr(self, var)))

    def read_restart(self, state, infile):
        attributes, variables = self.read_h5_restart(state, self.variables, infile)
        if attributes:
            self.nitts = attributes['nitts']
        if variables:
            for var, arr in variables.items():
                getattr(self, var)[...] = arr

    def write_restart(self, state, outfile):
        var_data = {key: getattr(self, key)
                    for key, var in self.variables.items() if var.write_to_restart}
        self.write_h5_restart(state, {'nitts': self.nitts}, self.variables, var_data, outfile)
