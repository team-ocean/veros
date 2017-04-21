import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method

RESTART_ATTRIBUTES = ("tempm1", "vtemp1", "saltm1", "vsalt1")

class TracerMonitor(VerosDiagnostic):
    def initialize(self, veros):
        self.tempm1 = 0.
        self.vtemp1 = 0.
        self.saltm1 = 0.
        self.vsalt1 = 0.

    def diagnose(self, veros):
        pass

    @veros_class_method
    def output(self, veros):
        """
        Diagnose tracer content
        """
        cell_volume = veros.area_t[2:-2, 2:-2, np.newaxis] * veros.dzt[np.newaxis, np.newaxis, :] \
                      * veros.maskT[2:-2, 2:-2, :]
        volm = np.sum(cell_volume)
        tempm = np.sum(cell_volume * veros.temp[2:-2, 2:-2, :, veros.tau])
        saltm = np.sum(cell_volume * veros.salt[2:-2, 2:-2, :, veros.tau])
        vtemp = np.sum(cell_volume * veros.temp[2:-2, 2:-2, :, veros.tau]**2)
        vsalt = np.sum(cell_volume * veros.salt[2:-2, 2:-2, :, veros.tau]**2)

        logging.warning(" mean temperature {} change to last {}".format(tempm/volm, (tempm-self.tempm1)/volm))
        logging.warning(" mean salinity    {} change to last {}".format(saltm/volm, (saltm-self.saltm1)/volm))
        logging.warning(" temperature var. {} change to last {}".format(vtemp/volm, (vtemp-self.vtemp1)/volm))
        logging.warning(" salinity var.    {} change to last {}".format(vsalt/volm, (vsalt-self.vsalt1)/volm))

        self.tempm1 = tempm
        self.vtemp1 = vtemp
        self.saltm1 = saltm
        self.vsalt1 = vsalt

    def read_restart(self, veros):
        attributes, variables = self.read_h5_restart(veros)
        for attr in RESTART_ATTRIBUTES:
            setattr(self, attr, attributes[attr])

    def write_restart(self, veros):
        attributes = {key: getattr(self, key) for key in RESTART_ATTRIBUTES}
        self.write_h5_restart(veros, attributes, {}, {})
