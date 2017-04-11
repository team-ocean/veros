import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method

class TracerMonitor(VerosDiagnostic):
    def initialize(self, veros):
        self.tempm1 = 0.
        self.vtemp1 = 0.
        self.saltm1 = 0.
        self.vsalt1 = 0.

    @veros_class_method
    def diagnose(self, veros):
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

        logging.warning("")
        logging.warning("mean temperature {} change to last {}".format(tempm/volm, (tempm-self.tempm1)/volm))
        logging.warning("mean salinity    {} change to last {}".format(saltm/volm, (saltm-self.saltm1)/volm))
        logging.warning("temperature var. {} change to last {}".format(vtemp/volm, (vtemp-self.vtemp1)/volm))
        logging.warning("salinity var.    {} change to last {}".format(vsalt/volm, (vsalt-self.vsalt1)/volm))

        self.tempm1 = tempm
        self.vtemp1 = vtemp
        self.saltm1 = saltm
        self.vsalt1 = vsalt

    def output(self, veros):
        pass

    def read_restart(self, veros):
        pass

    def write_restart(self, veros):
        pass
