import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method
import numpy as np

class NPZDMonitor(VerosDiagnostic):
    """Diagnostic monitoring nutrients and plankton concentrations
    """

    name = "npzd"
    output_frequency = None
    # restart_attributes = ("phytoplankton", "dic", "DIC_FLUX")
    restart_attributes = []
    output_variables = []
    show_npzd_graph = False

    def initialize(self, vs):
        self.phytoplankton_integrated = 0
        self.zooplankton_integrated = 0
        self.po4_integrated = 0
        self.po4 = 0
        self.detritus = 0

        if vs.enable_carbon:
            self.dic = np.zeros_like(vs.dic)
            self.caco3 = 0
            self.atmospheric_co2 = 0
            self.alkalinity = 0
            self.alkalinity_integrated = 0
            self.DIC_FLUX = 0
            self.dic_integrated = 0
            self.prca = 0

        if vs.enable_calcifiers:
            pass

        if vs.enable_nitrogen:
            pass

        if vs.enable_iron:
            pass

        if vs.enable_oxygen:
            pass

        # Whether or not to display a graph of the intended dynamics to the user
        # if self.show_npzd_graph:
        #     from graphviz import Digraph
        #     self.npzd_graph = Digraph("npzd_dynamics", filename="npzd_dynamics.gv")

        #     for name in vs.npzd_tracers:
        #         self.npzd_graph.attr('node', shape='square')
        #         self.npzd_graph.node(name)

        #     for rule in vs.npzd_rules:
        #         self.npzd_graph.append(rule[1], rule[2], label="?")


        #     self.npzd_graph.view()

    def diagnose(self, vs):
        pass

    @veros_class_method
    def output(self, vs):
        """Diagnose NPZD concentrations
        """


        cell_volume = vs.area_t[2:-2, 2:-2, np.newaxis] * vs.dzt[np.newaxis, np.newaxis, :]\
                * vs.maskT[2:-2, 2:-2, :]
        volm = np.sum(cell_volume)
        # po4_integrated = np.sum(cell_volume[vs.maskT] * vs.po4[2:-2, 2:-2, :][vs.maskT])
        # phytoplankton_integrated = np.sum(cell_volume * vs.phytoplankton[2:-2, 2:-2, :])
        # dic_integrated = np.sum(cell_volume[vs.maskT] * vs.dic[2:-2, 2:-2, :][vs.maskT])
        # alkalinity_integrated = np.sum(cell_volume * vs.alkalinity[2:-2, 2:-2, :])

        # logging.warning("Phytoplankton integrated {}, change to last {}".format(phytoplankton_integrated / volm, (phytoplankton_integrated - self.phytoplankton_integrated) / volm))

        # logging.warning("DIC integrated {}, change to last {}, min: {}, max: {}".format(dic_integrated / volm, (dic_integrated - self.dic_integrated) / volm, np.min(vs.dic), np.max(vs.dic)))

        # logging.warning("Average surface DIC: {}, min: {}, max: {}".format(vs.dic[:, :, -1].mean(), np.min((vs.dic[:, :, -1][2:-2, 2:-2])[vs.maskT[:, :, -1]]), np.max(vs.dic[:, :, -1])))


        # logging.warning("DIC Flux {}, mean change to last {}, min: {}, max: {}".format(vs.cflux[vs.maskT[:, :, -1]].mean(), (vs.cflux[vs.maskT[:, :, -1]].mean() - self.DIC_FLUX), np.min(vs.cflux[vs.maskT[:, :, -1]]), np.max(vs.cflux[vs.maskT[:, :, -1]])))

        # logging.warning("DIC Flux year mean: {}, min: {}, max:{}".format((vs.cflux * 60 * 60 * 24 * 365).mean(), vs.cflux.min() * 60 * 60 * 24 * 365, vs.cflux.max() * 60 * 60 * 24 * 365))

        # logging.warning("prca mean: {}, surface mean: {}".format(vs.prca.mean(), vs.prca[:-1].mean()))


        # self.phytoplankton_integrated = phytoplankton_integrated
        # self.dic_integrated = dic_integrated
        # self.po4_integrated = po4_integrated
        # self.DIC_FLUX = vs.cflux[vs.maskT[:, :, -1]].mean()
        # self.dic[...] = vs.dic[...]
        # self.prca = vs.prca.mean()
        # self.alkalinity_integrated = alkalinity_integrated


    def read_restart(self, vs):
        attributes, variables = self.read_h5_restart(vs)
        for attr in self.restart_attributes:
            setattr(self, attr, attributes[attr])

    def write_restart(self, vs, outfile):
        attributes = {key: getattr(self, key) for key in self.restart_attributes}
        self.write_h5_restart(vs, attributes, {}, {}, outfile)
