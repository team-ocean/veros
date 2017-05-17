from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from test_base import VerosUnitTest
from veros.core import idemix

class IdemixTest(VerosUnitTest):
    nx, ny, nz = 70, 60, 50
    extra_settings = {
                      "enable_idemix": True,
                      "enable_idemix_hor_diffusion": True,
                      "enable_idemix_superbee_advection": True,
                      "enable_idemix_upwind_advection": True,
                      "enable_eke": True,
                      "enable_store_cabbeling_heat": True,
                      "enable_eke_diss_bottom": True,
                      "enable_eke_diss_surfbot": True,
                      "enable_store_bottom_friction_tke": True,
                      "enable_TEM_friction": True,
                      }
    test_module = idemix

    def initialize(self):
        m = self.veros_legacy.main_module

        np.random.seed(123456789)
        for a in ("gamma","mu0","jstar","eke_diss_surfbot_frac","dt_tracer","tau_v","AB_eps"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt","dxu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.nx+4).astype(np.float))

        for a in ("dyt","dyu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.ny+4).astype(np.float))

        for a in ("cosu","cost"):
            self.set_attribute(a, 2*np.random.rand(self.ny+4)-1.)

        for a in ("zt","dzt","dzw"):
            self.set_attribute(a, np.random.rand(self.nz))

        for a in ("area_u", "area_v", "area_t", "coriolis_t", "forc_iw_bottom", "forc_iw_surface"):
            self.set_attribute(a, np.random.rand(self.nx+4, self.ny+4))

        for a in ("c0","v0","alpha_c","eke_diss_iw","K_diss_gm","K_diss_h","K_iso","K_gm","kappa_gm","P_diss_iso","P_diss_skew","P_diss_hmix","K_diss_bot"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("Nsqr","E_iw","dE_iw"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a,np.random.randint(0, 2, size=(self.nx+4,self.ny+4,self.nz)).astype(np.float))

        self.set_attribute("kbot",np.random.randint(0, self.nz, size=(self.nx+4,self.ny+4)))

        istemp = bool(np.random.randint(0,2))

        veros_args = (self.veros_new.temp,istemp,self.veros_new)
        veros_legacy_args = dict(is_=-1, ie_=m.nx+2, js_=-1, je_=m.ny+2, nz_=m.nz, tr=m.temp, istemp=istemp)

        self.test_routines = OrderedDict()
        self.test_routines["set_idemix_parameter"] = ((self.veros_new,), dict())
        self.test_routines["integrate_idemix"] = ((self.veros_new,), dict())


    def test_passed(self,routine):
        all_passed = True
        if routine == "set_idemix_parameter":
            for v in ("c0", "v0", "alpha_c",):
                passed = self.check_variable(v)
                if not passed:
                    all_passed = False
        elif routine == "integrate_idemix":
            for v in ("E_iw", "dE_iw", "iw_diss", "flux_east", "flux_north", "flux_top",):
                passed = self.check_variable(v)
                if not passed:
                    all_passed = False
        plt.show()
        return all_passed

if __name__ == "__main__":
    passed = IdemixTest().run()
    sys.exit(int(not passed))
