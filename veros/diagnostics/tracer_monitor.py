from veros import logger

from veros.variables import Variable
from veros.core.operators import numpy as npx
from veros.diagnostics.base import VerosDiagnostic
from veros.distributed import global_sum


class TracerMonitor(VerosDiagnostic):
    """Diagnostic monitoring global tracer contents / fluxes.

    Writes output to stdout (no binary output).
    """

    name = "tracer_monitor"
    supports_compilation = True
    output_frequency = None

    def __init__(self, state):
        self.var_meta = {
            "tempm1": Variable("tempm1", None, write_to_restart=True),
            "vtemp1": Variable("vtemp1", None, write_to_restart=True),
            "saltm1": Variable("saltm1", None, write_to_restart=True),
            "vsalt1": Variable("vsalt1", None, write_to_restart=True),
        }

    def initialize(self, state):
        self.initialize_variables(state)

    def diagnose(self, state):
        pass

    def finalize(self, state):
        """Return tracer statistics and update their previous values."""
        vs = state.variables
        tracer_vs = self.variables

        cell_volume = vs.area_t[2:-2, 2:-2, npx.newaxis] * vs.dzt[npx.newaxis, npx.newaxis, :] * vs.maskT[2:-2, 2:-2, :]
        volm = global_sum(npx.sum(cell_volume))
        tempm = global_sum(npx.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau]))
        saltm = global_sum(npx.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau]))
        vtemp = global_sum(npx.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau] ** 2))
        vsalt = global_sum(npx.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau] ** 2))

        values = dict(
            temp=tempm / volm,
            salt=saltm / volm,
            vtemp=vtemp / volm,
            vsalt=vsalt / volm,
            dtemp=(tempm - tracer_vs.tempm1) / volm,
            dsalt=(saltm - tracer_vs.saltm1) / volm,
            dvtemp=(vtemp - tracer_vs.vtemp1) / volm,
            dvsalt=(vsalt - tracer_vs.vsalt1) / volm,
        )

        tracer_vs.tempm1 = tempm
        tracer_vs.vtemp1 = vtemp
        tracer_vs.saltm1 = saltm
        tracer_vs.vsalt1 = vsalt
        return values

    def output(self, state):
        self.write_finalized(state, self.finalize(state))

    def write_finalized(self, state, values, time_value=None):
        logger.diagnostic(f" Mean temperature {values['temp']:.2e} change to last {values['dtemp']:.2e}")
        logger.diagnostic(f" Mean salinity    {values['salt']:.2e} change to last {values['dsalt']:.2e}")
        logger.diagnostic(f" Temperature var. {values['vtemp']:.2e} change to last {values['dvtemp']:.2e}")
        logger.diagnostic(f" Salinity var.    {values['vsalt']:.2e} change to last {values['dvsalt']:.2e}")
