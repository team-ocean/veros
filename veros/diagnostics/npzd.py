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
        npzd_sum = vs.detritus + vs.po4 + vs.zooplankton + vs.phytoplankton
        logging.warning("Total NPZD concentration: {}".format(npzd_sum.sum(axis=2)[5, 5],))
        npzd_sum *= vs.dzw[::-1]
        # logging.warning("Detritus: {}".format(vs.detritus[5, 5, :3]))
        # logging.warning("PO4: {}".format(vs.po4[5, 5, :3]))
        # logging.warning("Phytoplankton: {}".format(vs.phytoplankton[5, 5, :3]))
        # logging.warning("Zooplankton: {}".format(vs.zooplankton[5, 5, :3]))
        logging.warning("DOP - DON: {}".format(vs.dop[5,5,0] - vs.don[5,5,0]))
        logging.warning("Total NPZD organisms: {}".format(npzd_sum.sum(axis=2)[5, 5],))


    def read_restart(self, vs):
        attributes, variables = self.read_h5_restart(vs)
        for attr in self.restart_attributes:
            setattr(self, attr, attributes[attr])

    def write_restart(self, vs, outfile):
        attributes = {key: getattr(self, key) for key in self.restart_attributes}
        self.write_h5_restart(vs, attributes, {}, {}, outfile)
