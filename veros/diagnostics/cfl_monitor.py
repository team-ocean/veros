from veros import logger
from veros.core.operators import numpy as npx
from veros.diagnostics.base import VerosDiagnostic
from veros.distributed import global_max
from veros.core.geometry import cell_thickness
from veros.core.advection import wgrid_cfl_spacings


class CFLMonitor(VerosDiagnostic):
    """Diagnostic monitoring the maximum CFL number of the solution to detect
    instabilities.

    Writes output to stdout (no binary output).
    """

    name = "cfl_monitor"  #:
    supports_compilation = True
    output_frequency = None  #: Frequency (in seconds) in which output is written.

    def initialize(self, state):
        pass

    def diagnose(self, state):
        pass

    def finalize(self, state):
        """Return maximum horizontal and vertical CFL numbers."""
        vs = state.variables
        settings = state.settings

        cfl = global_max(
            npx.maximum(
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
                / cell_thickness(state)[2:-2, 2:-2]
                * settings.dt_tracer
            )
        )

        values = dict(cfl=cfl, wcfl=wcfl)

        if settings.enable_eke or settings.enable_tke or settings.enable_idemix:
            spacing_x, spacing_y, spacing_z = wgrid_cfl_spacings(state)
            cfl = global_max(
                npx.maximum(
                    npx.max(
                        npx.abs(vs.u_wgrid[2:-2, 2:-2, :])
                        * vs.maskU[2:-2, 2:-2, :]
                        / spacing_x[2:-2, 2:-2]
                        * settings.dt_tracer
                    ),
                    npx.max(
                        npx.abs(vs.v_wgrid[2:-2, 2:-2, :] * vs.cosu[npx.newaxis, 2:-2, npx.newaxis])
                        * vs.maskV[2:-2, 2:-2, :]
                        / spacing_y[2:-2, 2:-2]
                        * settings.dt_tracer
                    ),
                )
            )
            wcfl = global_max(
                npx.max(
                    npx.abs(vs.w_wgrid[2:-2, 2:-2, :])
                    * vs.maskW[2:-2, 2:-2, :]
                    / spacing_z[2:-2, 2:-2]
                    * settings.dt_tracer
                )
            )
            values.update(cfl_wgrid=cfl, wcfl_wgrid=wcfl)
        return values

    def output(self, state):
        self.write_finalized(state, self.finalize(state))

    def write_finalized(self, state, values, time_value=None):
        if not all(npx.isfinite(value) for value in values.values()):
            raise RuntimeError(f"CFL number is not finite at iteration {state.variables.itt}")
        logger.diagnostic(f" Maximal hor. CFL number = {values['cfl']}")
        logger.diagnostic(f" Maximal ver. CFL number = {values['wcfl']}")
        if "cfl_wgrid" in values:
            logger.diagnostic(f" Maximal hor. CFL number on w grid = {values['cfl_wgrid']}")
            logger.diagnostic(f" Maximal ver. CFL number on w grid = {values['wcfl_wgrid']}")
