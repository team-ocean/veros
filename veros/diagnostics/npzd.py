import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method
from graphviz import Digraph
import numpy as np

class NPZDMonitor(VerosDiagnostic):
    """Diagnostic monitoring nutrients and plankton concentrations
    """

    name = "npzd"
    output_frequency = None
    restart_attributes = []
    output_variables = []

    def initialize(self, vs):
        self.npzd_graph = Digraph("npzd_dynamics", filename="npzd_dynamics.gv", format="png")
        self.npzd_graph.graph_attr["splines"] = "ortho"
        self.npzd_graph.graph_attr["nodesep"] = "1"
        self.npzd_graph.graph_attr["node"] = "square"
        self.graph_saved = False


    def diagnose(self, vs):
        pass


    @veros_class_method
    def output(self, vs):
        """Print NPZD interaction graph
        """

        for tracer in vs.npzd_tracers:
            self.npzd_graph.node(tracer)

        for rule, source, sink, label in vs.npzd_rules:
            self.npzd_graph.edge(source, sink, xlabel=label)

        for rule, source, sink, label in vs.npzd_pre_rules:
            self.npzd_graph.edge(source, sink, xlabel=label, style="dotted")

        for rule, source, sink, label in vs.npzd_post_rules:
            self.npzd_graph.edge(source, sink, xlabel=label, style="dashed")

        if vs.sinking_speeds:
            self.npzd_graph.node("Bottom", shape="square")
            for sinker in vs.sinking_speeds:
                self.npzd_graph.edge(sinker, "Bottom", xlabel="sinking")

        if not self.graph_saved:
            self.graph_saved = True
            self.npzd_graph.view()


    def read_restart(self, vs):
        pass

    def write_restart(self, vs, outfile):
        pass
