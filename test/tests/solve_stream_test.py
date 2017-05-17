from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from test_base import VerosUnitTest
from veros.core import numerics, external

class StreamfunctionTest(VerosUnitTest):
    nx, ny, nz = 70, 60, 50
    first = True
    extra_settings = {
                        "enable_cyclic_x": True,
                        "enable_congrad_verbose": False,
                        "congr_epsilon": 1e-12,
                        "congr_max_iterations": 10000,
                     }
    def initialize(self):
        m = self.veros_legacy.main_module

        #np.random.seed(123456)
        for a in ("dt_mom", "AB_eps", "x_origin", "y_origin"):
            self.set_attribute(a,np.random.rand())

        for a in ("dxt",):
            self.set_attribute(a,100 * np.ones(self.nx+4) + np.random.rand(self.nx+4))

        for a in ("dyt",):
            self.set_attribute(a,100 * np.ones(self.ny+4) + np.random.rand(self.ny+4))

        for a in ("dzt",):
            self.set_attribute(a,10 + np.random.rand(self.nz))

        for a in ("psi", "dpsi"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,3))

        for a in ("du_mix", "dv_mix"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("u","v","du","dv","rho"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        kbot = np.random.randint(1, self.nz, size=(self.nx+4,self.ny+4))
        # add some islands, but avoid boundaries
        kbot[3:-3,3:-3].flat[np.random.randint(0, (self.nx-2) * (self.ny-2), size=10)] = 0
        self.set_attribute("kbot",kbot)

        for r in ("calc_grid", "calc_topo"):
            num_new, num_legacy = self.get_routine(r,submodule=numerics)
            num_new(self.veros_new)
            num_legacy()

        if self.first:
            init_new, init_legacy = self.get_routine("streamfunction_init",submodule=external)
            init_new(self.veros_new)
            init_legacy()
            self.first = False

        self.test_module = external
        veros_args = (self.veros_new,)
        veros_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["solve_streamfunction"] = (veros_args, veros_legacy_args)


    def test_passed(self,routine):
        all_passed = True
        for f in ("line_psin","psin","p_hydro","psi","dpsi", "du", "dv", "dpsin",
                  "u", "v"):
            passed = self.check_variable(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

if __name__ == "__main__":
    passed = StreamfunctionTest().run()
    sys.exit(int(not passed))
