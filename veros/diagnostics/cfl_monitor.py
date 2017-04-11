import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method

class CFLMonitor(VerosDiagnostic):
    def initialize(self, veros):
        pass

    @veros_class_method
    def diagnose(self, veros):
        """
        check for CFL violation
        """
        cfl = max(
            np.max(np.abs(veros.u[2:-2,2:-2,:,veros.tau]) * veros.maskU[2:-2,2:-2,:] \
                    / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                    * veros.dt_tracer),
            np.max(np.abs(veros.v[2:-2,2:-2,:,veros.tau]) * veros.maskV[2:-2,2:-2,:] \
                    / veros.dyt[np.newaxis, 2:-2, np.newaxis] * veros.dt_tracer)
        )
        wcfl = np.max(np.abs(veros.w[2:-2, 2:-2, :, veros.tau]) * veros.maskW[2:-2, 2:-2, :] \
                      / veros.dzt[np.newaxis, np.newaxis, :] * veros.dt_tracer)

        if np.isnan(cfl) or np.isnan(wcfl):
            raise RuntimeError("CFL number is NaN at iteration {}".format(veros.itt))

        logging.warning("maximal hor. CFL number = {}".format(cfl))
        logging.warning("maximal ver. CFL number = {}".format(wcfl))

        if veros.enable_eke or veros.enable_tke or veros.enable_idemix:
            cfl = max(
                np.max(np.abs(veros.u_wgrid[2:-2,2:-2,:]) * veros.maskU[2:-2,2:-2,:] \
                        / (veros.cost[np.newaxis, 2:-2, np.newaxis] * veros.dxt[2:-2, np.newaxis, np.newaxis]) \
                        * veros.dt_tracer),
                np.max(np.abs(veros.v_wgrid[2:-2,2:-2,:]) * veros.maskV[2:-2,2:-2,:] \
                        / veros.dyt[np.newaxis, 2:-2, np.newaxis] * veros.dt_tracer)
            )
            wcfl = np.max(np.abs(veros.w_wgrid[2:-2, 2:-2, :]) * veros.maskW[2:-2, 2:-2, :] \
                          / veros.dzt[np.newaxis, np.newaxis, :] * veros.dt_tracer)
            logging.warning("maximal hor. CFL number on w grid = {}".format(cfl))
            logging.warning("maximal ver. CFL number on w grid = {}".format(wcfl))

    def output(self, veros):
        pass

    def read_restart(self, veros):
        pass

    def write_restart(self, veros):
        pass
