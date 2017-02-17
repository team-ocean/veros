from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom import diffusion

class DiffusionTest(PyOMTest):
    extra_settings = {
                        "enable_cyclic_x": True,
                        "enable_conserve_energy": False,
                        "enable_hor_friction_cos_scaling": True,
                        "enable_tempsalt_sources": True,
                     }
    def initialize(self):
        m = self.pyom_legacy.main_module

        #np.random.seed(123456)
        self.set_attribute("hor_friction_cosPower", np.random.randint(1,5))

        for a in ("dt_tracer", "K_hbi", "K_h"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt","dxu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.nx+4).astype(np.float))

        for a in ("dyt","dyu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.ny+4).astype(np.float))

        for a in ("cosu","cost"):
            self.set_attribute(a,2*np.random.rand(self.ny+4)-1.)

        for a in ("dzt","dzw"):
            self.set_attribute(a,100*np.random.rand(self.nz))

        for a in ("flux_east","flux_north","flux_top","dtemp_hmix","dsalt_hmix","temp_source","salt_source"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("temp","salt","int_drhodS","int_drhodT"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a,np.random.randint(0,2,size=(self.nx+4,self.ny+4,self.nz)).astype(np.float))

        self.set_attribute("kbot",np.random.randint(0, self.nz, size=(self.nx+4,self.ny+4)))

        self.test_module = diffusion
        pyom_args = (self.pyom_new,)
        pyom_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines.update(
                              tempsalt_biharmonic = (pyom_args, pyom_legacy_args),
                              tempsalt_diffusion = (pyom_args, pyom_legacy_args),
                              tempsalt_sources = (pyom_args, pyom_legacy_args),
                             )

    def test_passed(self,routine):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","temp","salt","P_diss_hmix","dtemp_hmix","dsalt_hmix","P_diss_sources"):
            passed = self._check_var(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

    def _normalize(self,*arrays):
        norm = np.abs(arrays[0]).max()
        if norm == 0.:
            return arrays
        return (a / norm for a in arrays)

    def _check_var(self,var):
        v1, v2 = self.get_attribute(var)
        if v1.ndim > 1:
            v1 = v1[2:-2, 2:-2, ...]
        if v2.ndim > 1:
            v2 = v2[2:-2, 2:-2, ...]
        if v1 is None or v2 is None:
            raise RuntimeError(var)
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
    test = DiffusionTest(50, 25, 50, fortran=sys.argv[1])
    passed = test.run()
