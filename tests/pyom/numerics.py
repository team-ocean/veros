from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom.core import numerics

class NumericsTest(PyOMTest):
    repetitions = 1
    extra_settings = {
                        "enable_cyclic_x": True,
                        "coord_degree": False,
                     }
    def initialize(self):
        m = self.pyom_legacy.main_module

        #np.random.seed(123456)
        for a in ("x_origin", "y_origin"):
            self.set_attribute(a,np.random.rand())

        for a in ("dxt","dxu","xt","xu"):
            self.set_attribute(a,1 + 100*np.random.rand(self.nx+4))

        for a in ("dyt","dyu","yt","yu"):
            self.set_attribute(a,1 + 100*np.random.rand(self.ny+4))

        for a in ("dzt","dzw","zw","zt"):
            self.set_attribute(a,np.random.rand(self.nz))

        for a in ("cosu","cost","tantr"):
            self.set_attribute(a,2*np.random.rand(self.ny+4)-1.)

        for a in ("coriolis_t", "area_u", "area_v", "area_t"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4))

        for a in ("salt", "temp"):
            self.set_attribute(a,np.random.rand(self.nx+4, self.ny+4, self.nz, 3))

        kbot = np.random.randint(0, self.nz, size=(self.nx+4,self.ny+4))
        self.set_attribute("kbot",kbot)

        self.test_module = numerics
        pyom_args = (self.pyom_new,)
        pyom_legacy_args = dict()
        self.test_routines = OrderedDict()
        for r in ("calc_grid", "calc_topo", "calc_beta", "calc_initial_conditions"):
            self.test_routines[r] = (pyom_args, pyom_legacy_args)


    def test_passed(self,routine):
        all_passed = True
        for f in ("zt","zw","cosu","cost", "tantr", "area_u", "area_v", "area_t",
                  "beta", "xt", "xu", "dxu", "dxt", "yt", "yu", "dyu", "dyt", "dzt",
                  "dzw", "rho", "salt", "temp", "Nsqr", "Hd", "int_drhodT", "int_drhodS",
                  "ht", "hu", "hv", "hur", "hvr", "maskT", "maskW", "maskU", "maskV", "kbot"):
            passed = self._check_var(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

    def _normalize(self,*arrays):
        if any(a.size == 0 for a in arrays):
            return arrays
        norm = np.abs(arrays[0]).max()
        if norm == 0.:
            return arrays
        return (a / norm for a in arrays)

    def _check_var(self,var):
        v1, v2 = self.get_attribute(var)
        if v1 is None or v2 is None:
            raise RuntimeError(var)
        if v1.ndim > 1:
            v1 = v1[2:-2, 2:-2, ...]
        if v2.ndim > 1:
            v2 = v2[2:-2, 2:-2, ...]

        passed = np.allclose(*self._normalize(v1,v2))
        if not passed:
            print(var, np.abs(v1-v2).max(), v1.max(), v2.max(), np.where(v1 != v2))
            while v1.ndim > 2:
                v1 = v1[...,-1]
            while v2.ndim > 2:
                v2 = v2[...,-1]
            if v1.ndim == 2:
                fig, axes = plt.subplots(1,3)
                axes[0].imshow(v1)
                axes[0].set_title("New")
                axes[1].imshow(v2)
                axes[1].set_title("Legacy")
                axes[2].imshow(v1 - v2)
                axes[2].set_title("diff")
                fig.suptitle(var)
        return passed

if __name__ == "__main__":
    test = NumericsTest(150, 120, 50, fortran=sys.argv[1])
    passed = test.run()
