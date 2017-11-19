from collections import OrderedDict
import numpy as np
import sys

from test_base import VerosUnitTest
from veros.core import tke

class TKETest(VerosUnitTest):
    nx, ny, nz = 70, 60, 50
    extra_settings = {
                        "enable_cyclic_x": True,
                        "enable_idemix": True,
                        "tke_mxl_choice": 2,
                        "enable_tke": True,
                        "enable_eke": True,
                        "enable_store_cabbeling_heat": True,
                        "enable_store_bottom_friction_tke": False,
                        "enable_tke_hor_diffusion": True,
                        "enable_tke_superbee_advection": True,
                        "enable_tke_upwind_advection": True,
                     }
    def initialize(self):
        m = self.veros_legacy.main_module

        #np.random.seed(123456)

        for a in ("mxl_min","kappaM_0","kappaH_0","dt_tke", "c_k", "kappaM_min", "kappaM_max",
                  "K_h_tke","AB_eps","c_eps", "alpha_tke","dt_mom"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt","dxu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.nx+4).astype(np.float))

        for a in ("dyt","dyu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.ny+4).astype(np.float))

        for a in ("cosu","cost"):
            self.set_attribute(a,2*np.random.rand(self.ny+4)-1.)

        for a in ("dzt","dzw","zw"):
            self.set_attribute(a,100*np.random.rand(self.nz))

        for a in ("tke_surf_corr","ht","forc_tke_surface"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4))

        for a in ("K_diss_v", "P_diss_v", "P_diss_adv", "P_diss_nonlin", "eke_diss_tke",
                  "kappaM", "kappaH", "K_diss_bot", "eke_diss_iw", "K_diss_gm", "K_diss_h", "tke_diss",
                  "P_diss_skew", "P_diss_hmix", "P_diss_iso","alpha_c","mxl","iw_diss", "Prandtlnumber"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("tke","dtke","Nsqr","E_iw"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a,np.random.randint(0,2,size=(self.nx+4,self.ny+4,self.nz)).astype(np.float))

        self.set_attribute("kbot",np.random.randint(0, self.nz, size=(self.nx+4,self.ny+4)))

        self.test_module = tke
        veros_args = (self.veros_new,)
        veros_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["set_tke_diffusivities"] = (veros_args, veros_legacy_args)
        self.test_routines["integrate_tke"] = (veros_args, veros_legacy_args)

    def test_passed(self,routine):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","tke","dtke","tke_surf_corr","tke_diss",
                  "kappaH","kappaM","Prandtlnumber","mxl","sqrttke"):
            passed = self.check_variable(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

if __name__ == "__main__":
    passed = TKETest().run()
    sys.exit(int(not passed))
