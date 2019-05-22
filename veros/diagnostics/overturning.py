from collections import OrderedDict
import os

from loguru import logger

from .. import veros_method
from .diagnostic import VerosDiagnostic
from ..core import density
from ..variables import Variable, allocate
from ..distributed import global_sum


SIGMA = Variable(
    'Sigma axis', ('sigma',), 'kg/m^3', 'Sigma axis', output=True,
    time_dependent=False, write_to_restart=True
)

OVERTURNING_VARIABLES = OrderedDict([
    ('trans', Variable(
        'Meridional transport', ('yu', 'sigma'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    )),
    ('vsf_iso', Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    )),
    ('vsf_depth', Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    )),
])
ISONEUTRAL_VARIABLES = OrderedDict([
    ('bolus_iso', Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    )),
    ('bolus_depth', Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    )),
])


class Overturning(VerosDiagnostic):
    """Isopycnal overturning diagnostic. Computes and writes vertical streamfunctions
    (zonally averaged).
    """

    name = 'overturning'  #:
    output_path = '{identifier}.overturning.nc'  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.
    p_ref = 2000.  #: Reference pressure for isopycnals

    def __init__(self, vs):
        self.sigma_var = SIGMA
        self.mean_variables = OVERTURNING_VARIABLES
        if vs.enable_neutral_diffusion and vs.enable_skew_diffusion:
            self.mean_variables.update(ISONEUTRAL_VARIABLES)
        self.variables = self.mean_variables.copy()
        self.variables.update({'sigma': self.sigma_var})

    @veros_method
    def initialize(self, vs):
        self.nitts = 0
        self.nlevel = vs.nz * 4
        self._allocate(vs)

        # sigma levels
        self.sige = float(density.get_rho(vs, 35., -2., self.p_ref))
        self.sigs = float(density.get_rho(vs, 35., 30., self.p_ref))
        self.dsig = (self.sige - self.sigs) / (self.nlevel - 1)

        logger.debug(' Sigma ranges for overturning diagnostic:')
        logger.debug(' Start sigma0 = {:.1f}'.format(self.sigs))
        logger.debug(' End sigma0 = {:.1f}'.format(self.sige))
        logger.debug(' Delta sigma0 = {:.1e}'.format(self.dsig))
        if vs.enable_neutral_diffusion and vs.enable_skew_diffusion:
            logger.debug(' Also calculating overturning by eddy-driven velocities')

        self.sigma[...] = self.sigs + self.dsig * np.arange(self.nlevel)

        # precalculate area below z levels
        self.zarea[2:-2, :] = np.cumsum(global_sum(vs, np.sum(
            vs.dxt[2:-2, np.newaxis, np.newaxis]
            * vs.cosu[np.newaxis, 2:-2, np.newaxis]
            * vs.maskV[2:-2, 2:-2, :], axis=0)) * vs.dzt[np.newaxis, :], axis=1)

        self.initialize_output(vs, self.variables,
                               var_data={'sigma': self.sigma},
                               extra_dimensions={'sigma': self.nlevel})

    @veros_method
    def _allocate(self, vs):
        self.sigma = allocate(vs, (self.nlevel,))
        self.zarea = allocate(vs, ('yu', 'zt'))
        self.trans = allocate(vs, ('yu', self.nlevel))
        self.vsf_iso = allocate(vs, ('yu', 'zt'))
        self.vsf_depth = allocate(vs, ('yu', 'zt'))
        if vs.enable_neutral_diffusion and vs.enable_skew_diffusion:
            self.bolus_iso = allocate(vs, ('yu', 'zt'))
            self.bolus_depth = allocate(vs, ('yu', 'zt'))

    @veros_method
    def diagnose(self, vs):
        # sigma at p_ref
        sig_loc = allocate(vs, ('xt', 'yt', 'zt'))
        sig_loc[2:-2, 2:-1, :] = density.get_rho(vs,
                                                 vs.salt[2:-2, 2:-1, :, vs.tau],
                                                 vs.temp[2:-2, 2:-1, :, vs.tau],
                                                 self.p_ref)

        # transports below isopycnals and area below isopycnals
        sig_loc_face = 0.5 * (sig_loc[2:-2, 2:-2, :] + sig_loc[2:-2, 3:-1, :])
        trans = allocate(vs, ('yu', self.nlevel))
        z_sig = allocate(vs, ('yu', self.nlevel))

        for m in range(self.nlevel):
            # NOTE: vectorized version would be O(N^4) in memory
            # consider cythonizing if performance-critical
            mask = sig_loc_face > self.sigma[m]
            trans[2:-2, m] = global_sum(vs, np.sum(
                vs.v[2:-2, 2:-2, :, vs.tau]
                * vs.dxt[2:-2, np.newaxis, np.newaxis]
                * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                * vs.dzt[np.newaxis, np.newaxis, :]
                * vs.maskV[2:-2, 2:-2, :] * mask, axis=(0, 2)))
            z_sig[2:-2, m] = global_sum(vs, np.sum(
                vs.dzt[np.newaxis, np.newaxis, :]
                * vs.dxt[2:-2, np.newaxis, np.newaxis]
                * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                * vs.maskV[2:-2, 2:-2, :] * mask, axis=(0, 2)))
        self.trans += trans

        if vs.enable_neutral_diffusion and vs.enable_skew_diffusion:
            bolus_trans = allocate(vs, ('yu', self.nlevel))
            # eddy-driven transports below isopycnals
            for m in range(self.nlevel):
                # NOTE: see above
                mask = sig_loc_face > self.sigma[m]
                bolus_trans[2:-2, m] = global_sum(vs,
                    np.sum(
                        (vs.B1_gm[2:-2, 2:-2, 1:] - vs.B1_gm[2:-2, 2:-2, :-1])
                        * vs.dxt[2:-2, np.newaxis, np.newaxis]
                        * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                        * vs.maskV[2:-2, 2:-2, 1:]
                        * mask[:, :, 1:],
                        axis=(0, 2)
                    )
                    + np.sum(
                        vs.B1_gm[2:-2, 2:-2, 0]
                        * vs.dxt[2:-2, np.newaxis]
                        * vs.cosu[np.newaxis, 2:-2]
                        * vs.maskV[2:-2, 2:-2, 0]
                        * mask[:, :, 0],
                        axis=0
                    )
                )

        # streamfunction on geopotentials
        self.vsf_depth[2:-2, :] += np.cumsum(global_sum(vs, np.sum(
            vs.dxt[2:-2, np.newaxis, np.newaxis]
            * vs.cosu[np.newaxis, 2:-2, np.newaxis]
            * vs.v[2:-2, 2:-2, :, vs.tau]
            * vs.maskV[2:-2, 2:-2, :], axis=0)) * vs.dzt[np.newaxis, :], axis=1)

        if vs.enable_neutral_diffusion and vs.enable_skew_diffusion:
            # streamfunction for eddy driven velocity on geopotentials
            self.bolus_depth[2:-2, :] += global_sum(vs, np.sum(
                vs.dxt[2:-2, np.newaxis, np.newaxis]
                * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                * vs.B1_gm[2:-2, 2:-2, :], axis=0))
        # interpolate from isopycnals to depth
        self.vsf_iso[2:-2, :] += self._interpolate_along_axis(vs,
                                                              z_sig[2:-2, :], trans[2:-2, :],
                                                              self.zarea[2:-2, :], 1)
        if vs.enable_neutral_diffusion and vs.enable_skew_diffusion:
            self.bolus_iso[2:-2, :] += self._interpolate_along_axis(vs,
                                                                    z_sig[2:-2, :], bolus_trans[2:-2, :],
                                                                    self.zarea[2:-2, :], 1)

        self.nitts += 1


    @veros_method
    def _interpolate_along_axis(self, vs, coords, arr, interp_coords, axis=0):
        # TODO: clean up this mess

        if coords.ndim == 1:
            if len(coords) != arr.shape[axis]:
                raise ValueError('Coordinate shape must match array shape along axis')
        elif coords.ndim == arr.ndim:
            if coords.shape != arr.shape:
                raise ValueError('Coordinate shape must match array shape')
        else:
            raise ValueError('Coordinate shape must match array dimensions')

        if axis != 0:
            arr = np.moveaxis(arr, axis, 0)
            coords = np.moveaxis(coords, axis, 0)
            interp_coords = np.moveaxis(interp_coords, axis, 0)

        diff = coords[np.newaxis, :, ...] - interp_coords[:, np.newaxis, ...]
        diff_m = np.where(diff <= 0., np.abs(diff), np.inf)
        diff_p = np.where(diff > 0., np.abs(diff), np.inf)
        i_m = np.asarray(np.argmin(diff_m, axis=1))
        i_p = np.asarray(np.argmin(diff_p, axis=1))
        mask = np.all(np.isinf(diff_m), axis=1)
        i_m[mask] = i_p[mask]
        mask = np.all(np.isinf(diff_p), axis=1)
        i_p[mask] = i_m[mask]
        full_shape = (slice(None),) + (np.newaxis,) * (arr.ndim - 1)
        if coords.ndim == 1:
            i_p_full = i_p[full_shape] * np.ones(arr.shape)
            i_m_full = i_m[full_shape] * np.ones(arr.shape)
        else:
            i_p_full = i_p
            i_m_full = i_m
        ii = np.indices(i_p_full.shape)
        i_p_slice = (i_p_full,) + tuple(ii[1:])
        i_m_slice = (i_m_full,) + tuple(ii[1:])
        dx = (coords[i_p_slice] - coords[i_m_slice])
        pos = np.where(dx == 0., 0., (coords[i_p_slice] - interp_coords) / (dx + 1e-12))
        return np.moveaxis(arr[i_p_slice] * (1. - pos) + arr[i_m_slice] * pos, 0, axis)

    @veros_method
    def output(self, vs):
        if not os.path.isfile(self.get_output_file_name(vs)):
            self.initialize_output(vs, self.variables,
                                   var_data={'sigma': self.sigma},
                                   extra_dimensions={'sigma': self.nlevel})

        if self.nitts > 0:
            for var in self.mean_variables.keys():
                getattr(self, var)[...] *= 1. / self.nitts

        var_metadata = {key: var for key, var in self.mean_variables.items()
                        if var.output and var.time_dependent}
        var_data = {key: getattr(self, key) for key, var in self.mean_variables.items()
                    if var.output and var.time_dependent}
        self.write_output(vs, var_metadata, var_data)

        self.nitts = 0
        for var in self.mean_variables.keys():
            getattr(self, var)[...] = 0.

    @veros_method
    def read_restart(self, vs, infile):
        attributes, variables = self.read_h5_restart(vs, self.variables, infile)
        if attributes:
            self.nitts = attributes['nitts']
        if variables:
            for var, arr in variables.items():
                getattr(self, var)[...] = arr

    @veros_method
    def write_restart(self, vs, outfile):
        var_data = {key: getattr(self, key)
                    for key, var in self.variables.items() if var.write_to_restart}
        self.write_h5_restart(vs, {'nitts': self.nitts}, self.variables, var_data, outfile)
