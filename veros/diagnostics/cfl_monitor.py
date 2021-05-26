from veros import logger
from veros.core.operators import numpy as npx
from veros.diagnostics.base import VerosDiagnostic
from veros.distributed import global_max


class CFLMonitor(VerosDiagnostic):
    """Diagnostic monitoring the maximum CFL number of the solution to detect
    instabilities.

    Writes output to stdout (no binary output).
    """

    name = "cfl_monitor"  #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.

    def initialize(self, state):
        pass

    def diagnose(self, state):
        pass

    def output(self, state):
        """
        check for CFL violation
        """
        vs = state.variables
        settings = state.settings

        cfl = global_max(
            max(
                npx.max(
                    npx.abs(vs.u[2:-2, 2:-2, :, vs.tau])
                    * vs.maskU[2:-2, 2:-2, :]
                    / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
                    * settings.dt_tracer
                ),
                npx.max(
                    npx.abs(vs.v[2:-2, 2:-2, :, vs.tau])
                    * vs.maskV[2:-2, 2:-2, :]
                    / vs.dyt[npx.newaxis, 2:-2, npx.newaxis]
                    * settings.dt_tracer
                ),
            )
        )
        wcfl = global_max(
            npx.max(
                npx.abs(vs.w[2:-2, 2:-2, :, vs.tau])
                * vs.maskW[2:-2, 2:-2, :]
                / vs.dzt[npx.newaxis, npx.newaxis, :]
                * settings.dt_tracer
            )
        )

        if npx.isnan(cfl) or npx.isnan(wcfl):
            raise RuntimeError(f"CFL number is NaN at iteration {vs.itt}")

        logger.diagnostic(f" Maximal hor. CFL number = {cfl}")
        logger.diagnostic(f" Maximal ver. CFL number = {wcfl}")

        if settings.enable_eke or settings.enable_tke or settings.enable_idemix:
            cfl = global_max(
                max(
                    npx.max(
                        npx.abs(vs.u_wgrid[2:-2, 2:-2, :])
                        * vs.maskU[2:-2, 2:-2, :]
                        / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
                        * settings.dt_tracer
                    ),
                    npx.max(
                        npx.abs(vs.v_wgrid[2:-2, 2:-2, :])
                        * vs.maskV[2:-2, 2:-2, :]
                        / vs.dyt[npx.newaxis, 2:-2, npx.newaxis]
                        * settings.dt_tracer
                    ),
                )
            )
            wcfl = global_max(
                npx.max(
                    npx.abs(vs.w_wgrid[2:-2, 2:-2, :])
                    * vs.maskW[2:-2, 2:-2, :]
                    / vs.dzt[npx.newaxis, npx.newaxis, :]
                    * settings.dt_tracer
                )
            )
            logger.diagnostic(f" Maximal hor. CFL number on w grid = {cfl}")
            logger.diagnostic(f" Maximal ver. CFL number on w grid = {wcfl}")
