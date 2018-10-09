import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method

class NPZDMonitor(VerosDiagnostic):
    """Diagnostic monitoring nutrients and plankton concentrations
    """

    name = "npzd"
    output_frequency = None
    restart_attributes = ("phytoplankton", "zooplankton", "po4", "detritus")

    def initialize(self, vs):
        self.phytoplankton = 0
        self.zooplankton = 0
        self.po4 = 0
        self.detritus = 0

    def diagnose(self, vs):
        pass

    @veros_class_method
    def output(self, vs):
        """Diagnose NPZD concentrations
        """
        x = 50
        y = 24
        npzd_sum = vs.detritus + vs.po4 + vs.zooplankton + vs.phytoplankton + vs.dop + vs.don + vs.no3 + vs.diazotroph
        # logging.warning("Total NPZD concentration: {}".format(npzd_sum.sum(axis=2)[x, y],))
        logging.warning("Total NPZD concentration: {}".format(npzd_sum.sum()))
        npzd_sum *= vs.dzw[::-1]
        logging.warning("PO4: {}".format(vs.po4[x, y, -4:]))
        logging.warning("DOP: {}".format(vs.dop[x, y, -4:]))
        logging.warning("DON: {}".format(vs.don[x, y, -4:]))
        logging.warning("NO3: {}".format(vs.no3[x, y, -4:]))
        logging.warning("phytoplankton: {}".format(vs.phytoplankton[x, y, -4:]))
        logging.warning("detritus: {}".format(vs.detritus[x, y, -4:]))
        logging.warning("zooplankton: {}".format(vs.zooplankton[x, y, -4:]))
        logging.warning("diazotroph: {}".format(vs.diazotroph[x, y, -4:]))
        # logging.warning("maskT: {}".format(vs.maskT[x, y]))
        # logging.warning("dz: {}".format(vs.dzt))


    def read_restart(self, vs):
        attributes, variables = self.read_h5_restart(vs)
        for attr in self.restart_attributes:
            setattr(self, attr, attributes[attr])

    def write_restart(self, vs, outfile):
        attributes = {key: getattr(self, key) for key in self.restart_attributes}
        self.write_h5_restart(vs, attributes, {}, {}, outfile)
