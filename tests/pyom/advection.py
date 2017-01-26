import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom import advection

class AdvectionTest(PyOMTest):
    def initialize(self):
        m = self.pyom_legacy.main_module

        np.random.seed(123456)
        self.set_attribute("dt_tracer", 3600.)

        for a in ("dxt",):
            self.set_attribute(a,np.random.randn(self.nx+4))

        for a in ("cosu","cost","dyt"):
            self.set_attribute(a,np.random.randn(self.ny+4))

        for a in ("dzt",):
            self.set_attribute(a,np.random.randn(self.nz))

        for a in ("flux_east","flux_north","flux_top"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("u","v","w","Hd"):
            self.set_attribute(a,np.ones((self.nx+4,self.ny+4,self.nz,3)))#np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a,np.random.randint(0,2,size=(self.nx+4,self.ny+4,self.nz)).astype(np.float))

        self.test_module = advection
        pyom_args = (self.pyom_new.flux_east, self.pyom_new.flux_north, self.pyom_new.flux_top, self.pyom_new.Hd[...,1], self.pyom_new)
        pyom_legacy_args = dict(is_=-1, ie_=m.nx+2, js_=-1, je_=m.ny+2, nz_=m.nz, adv_fe=m.flux_east, adv_fn=m.flux_north, adv_ft=m.flux_top, var=m.Hd[...,1])
        self.test_routines = (("adv_flux_2nd", pyom_args, pyom_legacy_args),
                              ("adv_flux_superbee", pyom_args, pyom_legacy_args),
                              ("adv_flux_upwind_wgrid", pyom_args, pyom_legacy_args),
                              ("calculate_velocity_on_wgrid", pyom_args, pyom_legacy_args),
                              ("adv_flux_superbee_wgrid", pyom_args, pyom_legacy_args)
                              )

    def test_passed(self):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","Hd"):
            v1, v2 = self.get_attribute(f)
            passed = np.allclose(v1, v2)
            if not passed:
                fig, axes = plt.subplots(1,3)
                axes[0].imshow(v1[...,0])
                axes[1].imshow(v2[...,0])
                axes[2].imshow(v1[...,0] - v2[...,0])
                print(f, v1.max(), v2.max())
                all_passed = False
        plt.show()
        return all_passed


if __name__ == "__main__":
    test = AdvectionTest(150, 70, 50, fortran=sys.argv[1])
    passed = test.run()
