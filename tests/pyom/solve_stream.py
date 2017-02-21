from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom import numerics, external

class StreamfunctionTest(PyOMTest):
    extra_settings = {
                        "enable_cyclic_x": False,
                        "enable_congrad_verbose": False,
                        "congr_epsilon": 1e-12,
                        "congr_max_iterations": 10000,
                     }
    def initialize(self):
        m = self.pyom_legacy.main_module

        np.random.seed(123456)
        for a in ("dt_mom", "AB_eps", "x_origin", "y_origin"):
            self.set_attribute(a,np.random.rand())

        for a in ("dxt",):
            self.set_attribute(a,1 + 100*np.random.rand(self.nx+4))

        for a in ("dyt",):
            self.set_attribute(a,1 + 100*np.random.rand(self.ny+4))

        for a in ("dzt",):
            self.set_attribute(a,np.random.rand(self.nz))

        for a in ("psi", "dpsi"):
            self.set_attribute(a,np.random.rand(self.nx+4,self.ny+4, 3))

        for a in ("du_mix", "dv_mix"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("u","v","du","dv"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        kbot = np.random.randint(1, self.nz, size=(self.nx+4,self.ny+4))
        kbot.flat[np.random.randint(0, np.prod(kbot.shape), size=50)] = 0
        self.set_attribute("kbot",kbot)

        for r in ("calc_grid", "calc_topo"):
            num_new, num_legacy = self.get_routine(r,submodule=numerics)
            num_new(self.pyom_new)
            num_legacy()

        self.test_module = external
        pyom_args = (self.pyom_new,)
        pyom_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["streamfunction_init"] = (pyom_args, pyom_legacy_args)
        self.test_routines["solve_streamfunction"] = (pyom_args, pyom_legacy_args)


    def test_passed(self,routine):
        all_passed = True
        for f in ("line_psin","psin","p_hydro","psi","dpsi", "du", "dv", "dpsin",
                  "u", "v"):
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
        #if v1.ndim > 1:
        #    v1 = v1[2:-2, 2:-2, ...]
        #if v2.ndim > 1:
        #    v2 = v2[2:-2, 2:-2, ...]

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
    test = StreamfunctionTest(150, 200, 50, fortran=sys.argv[1])
    passed = test.run()
