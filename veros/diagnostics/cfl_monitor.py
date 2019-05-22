from loguru import logger

from .diagnostic import VerosDiagnostic
from .. import veros_method
from ..distributed import global_max


class CFLMonitor(VerosDiagnostic):
    """Diagnostic monitoring the maximum CFL number of the solution to detect
    instabilities.

    Writes output to stdout (no binary output).
    """
    name = 'cfl_monitor' #:
    output_frequency = None  # :Frequency (in seconds) in which output is written.

    def initialize(self, vs):
        pass

    def diagnose(self, vs):
        pass

    @veros_method
    def output(self, vs):
        """
        check for CFL violation
        """
        cfl = global_max(vs, max(
            np.max(np.abs(vs.u[2:-2, 2:-2, :, vs.tau]) * vs.maskU[2:-2, 2:-2, :]
                   / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                   * vs.dt_tracer),
            np.max(np.abs(vs.v[2:-2, 2:-2, :, vs.tau]) * vs.maskV[2:-2, 2:-2, :]
                   / vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.dt_tracer)
        ))
        wcfl = global_max(vs, np.max(
            np.abs(vs.w[2:-2, 2:-2, :, vs.tau]) * vs.maskW[2:-2, 2:-2, :]
                      / vs.dzt[np.newaxis, np.newaxis, :] * vs.dt_tracer
        ))

        if np.isnan(cfl) or np.isnan(wcfl):
            raise RuntimeError('CFL number is NaN at iteration {}'.format(vs.itt))

        logger.warning(' Maximal hor. CFL number = {}'.format(float(cfl)))
        logger.warning(' Maximal ver. CFL number = {}'.format(float(wcfl)))

        if vs.enable_eke or vs.enable_tke or vs.enable_idemix:
            cfl = global_max(vs, max(
                np.max(np.abs(vs.u_wgrid[2:-2, 2:-2, :]) * vs.maskU[2:-2, 2:-2, :]
                       / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                       * vs.dt_tracer),
                np.max(np.abs(vs.v_wgrid[2:-2, 2:-2, :]) * vs.maskV[2:-2, 2:-2, :]
                       / vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.dt_tracer)
            ))
            wcfl = global_max(vs, np.max(
                np.abs(vs.w_wgrid[2:-2, 2:-2, :]) * vs.maskW[2:-2, 2:-2, :]
                    / vs.dzt[np.newaxis, np.newaxis, :] * vs.dt_tracer
            ))
            logger.warning(' Maximal hor. CFL number on w grid = {}'.format(float(cfl)))
            logger.warning(' Maximal ver. CFL number on w grid = {}'.format(float(wcfl)))

    def read_restart(self, vs, infile):
        pass

    def write_restart(self, vs, outfile):
        pass
