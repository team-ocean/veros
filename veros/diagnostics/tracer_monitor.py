from loguru import logger

from .diagnostic import VerosDiagnostic
from .. import veros_method
from ..distributed import global_sum


class TracerMonitor(VerosDiagnostic):
    """Diagnostic monitoring global tracer contents / fluxes.

    Writes output to stdout (no binary output).
    """
    name = 'tracer_monitor' #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    #: internal attributes to write to restart file
    restart_attributes = ('tempm1', 'vtemp1', 'saltm1', 'vsalt1')

    def initialize(self, vs):
        self.tempm1 = 0.
        self.vtemp1 = 0.
        self.saltm1 = 0.
        self.vsalt1 = 0.

    def diagnose(self, vs):
        pass

    @veros_method
    def output(self, vs):
        """
        Diagnose tracer content
        """
        cell_volume = vs.area_t[2:-2, 2:-2, np.newaxis] * vs.dzt[np.newaxis, np.newaxis, :] \
            * vs.maskT[2:-2, 2:-2, :]
        volm = global_sum(vs, np.sum(cell_volume))
        tempm = global_sum(vs, np.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau]))
        saltm = global_sum(vs, np.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau]))
        vtemp = global_sum(vs, np.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau]**2))
        vsalt = global_sum(vs, np.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau]**2))

        logger.warning(' Mean temperature {} change to last {}'
                       .format(float(tempm / volm), float((tempm - self.tempm1) / volm)))
        logger.warning(' Mean salinity    {} change to last {}'
                       .format(float(saltm / volm), float((saltm - self.saltm1) / volm)))
        logger.warning(' Temperature var. {} change to last {}'
                       .format(float(vtemp / volm), float((vtemp - self.vtemp1) / volm)))
        logger.warning(' Salinity var.    {} change to last {}'
                       .format(float(vsalt / volm), float((vsalt - self.vsalt1) / volm)))

        self.tempm1 = tempm
        self.vtemp1 = vtemp
        self.saltm1 = saltm
        self.vsalt1 = vsalt

    def read_restart(self, vs, infile):
        attributes, variables = self.read_h5_restart(vs, {}, infile)
        for attr in self.restart_attributes:
            setattr(self, attr, attributes[attr])

    def write_restart(self, vs, outfile):
        attributes = {key: getattr(self, key) for key in self.restart_attributes}
        self.write_h5_restart(vs, attributes, {}, {}, outfile)
