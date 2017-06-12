import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method


class TracerMonitor(VerosDiagnostic):
    """Diagnostic monitoring global tracer contents / fluxes.

    Writes output to stdout (no binary output).
    """
    name = "tracer_monitor" #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    #: internal attributes to write to restart file
    restart_attributes = ("tempm1", "vtemp1", "saltm1", "vsalt1")

    def initialize(self, vs):
        self.tempm1 = 0.
        self.vtemp1 = 0.
        self.saltm1 = 0.
        self.vsalt1 = 0.

    def diagnose(self, vs):
        pass

    @veros_class_method
    def output(self, vs):
        """
        Diagnose tracer content
        """
        cell_volume = vs.area_t[2:-2, 2:-2, np.newaxis] * vs.dzt[np.newaxis, np.newaxis, :] \
            * vs.maskT[2:-2, 2:-2, :]
        volm = np.sum(cell_volume)
        tempm = np.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau])
        saltm = np.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau])
        vtemp = np.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau]**2)
        vsalt = np.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau]**2)

        logging.warning(" mean temperature {} change to last {}"
                        .format(tempm / volm, (tempm - self.tempm1) / volm))
        logging.warning(" mean salinity    {} change to last {}"
                        .format(saltm / volm, (saltm - self.saltm1) / volm))
        logging.warning(" temperature var. {} change to last {}"
                        .format(vtemp / volm, (vtemp - self.vtemp1) / volm))
        logging.warning(" salinity var.    {} change to last {}"
                        .format(vsalt / volm, (vsalt - self.vsalt1) / volm))

        self.tempm1 = tempm
        self.vtemp1 = vtemp
        self.saltm1 = saltm
        self.vsalt1 = vsalt

    def read_restart(self, vs):
        attributes, variables = self.read_h5_restart(vs)
        for attr in self.restart_attributes:
            setattr(self, attr, attributes[attr])

    def write_restart(self, vs, outfile):
        attributes = {key: getattr(self, key) for key in self.restart_attributes}
        self.write_h5_restart(vs, attributes, {}, {}, outfile)
