from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from test_base import VerosUnitTest
from veros.core import advection, numerics

class AdvectionTest(VerosUnitTest):
    nx, ny, nz = 70, 60, 50

    def initialize(self):
        m = self.veros_legacy.main_module

        np.random.seed(123456)
        self.set_attribute("dt_tracer", 3600.)

        for a in ("dxt",):
            self.set_attribute(a,np.random.randint(1,100,size=self.nx+4).astype(np.float))

        for a in ("dyt",):
            self.set_attribute(a,np.random.randint(1,100,size=self.ny+4).astype(np.float))

        for a in ("cosu","cost"):
            self.set_attribute(a,2*np.random.rand(self.ny+4)-1.)

        for a in ("dzt","dzw"):
            self.set_attribute(a,100*np.random.rand(self.nz))

        for a in ("flux_east","flux_north","flux_top","u_wgrid","v_wgrid","w_wgrid"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("u","v","w","Hd"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        self.set_attribute("kbot",np.random.randint(0,self.nz,size=(self.nx+4,self.ny+4)).astype(np.float))
        calc_topo_new, calc_topo_legacy = self.get_routine("calc_topo",submodule=numerics)
        calc_topo_new(self.veros_new)
        calc_topo_legacy()

        self.test_module = advection
        veros_args = (self.veros_new, self.veros_new.flux_east, self.veros_new.flux_north, self.veros_new.flux_top, self.veros_new.Hd[...,1])
        veros_legacy_args = dict(is_=-1, ie_=m.nx+2, js_=-1, je_=m.ny+2, nz_=m.nz, adv_fe=m.flux_east, adv_fn=m.flux_north, adv_ft=m.flux_top, var=m.Hd[...,1])
        self.test_routines = OrderedDict()
        self.test_routines["calculate_velocity_on_wgrid"] = ((self.veros_new,), dict())
        self.test_routines.update(
                              adv_flux_2nd = (veros_args, veros_legacy_args),
                              adv_flux_superbee = (veros_args, veros_legacy_args),
                              adv_flux_upwind_wgrid = (veros_args, veros_legacy_args),
                              adv_flux_superbee_wgrid = (veros_args, veros_legacy_args)
                             )

    def test_passed(self,routine):
        all_passed = True
        if routine == "calculate_velocity_on_wgrid":
            for v in ("u_wgrid", "v_wgrid", "w_wgrid"):
                passed = self.check_variable(v)
                if not passed:
                    all_passed = False
        for f in ("flux_east","flux_north","flux_top"):
            passed = self.check_variable(f)
            if not passed:
                all_passed = False
        #plt.show()
        return all_passed

if __name__ == "__main__":
    passed = AdvectionTest().run()
    sys.exit(int(not passed))
