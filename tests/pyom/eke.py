from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom import eke

class EKETest(PyOMTest):
    extra_settings = {
                        "enable_cyclic_x": True,
                        "enable_eke_leewave_dissipation": True,
                        "enable_eke": True,
                        "enable_TEM_friction": True,
                        "enable_eke_isopycnal_diffusion": True,
                        "enable_store_cabbeling_heat": False,
                        "enable_eke_superbee_advection": True,
                        "enable_eke_upwind_advection": True
                     }
    def initialize(self):
        m = self.pyom_legacy.main_module

        np.random.seed(123456)

        for a in ("eke_hrms_k0_min","eke_k_max","eke_c_k","eke_crhin","eke_cross",
                  "eke_lmin","K_gm_0","K_iso_0","c_lee0","eke_Ri0","eke_Ri1","eke_int_diss0",
                  "kappa_EKE0","eke_r_bot","eke_c_eps","alpha_eke","dt_tracer","AB_eps"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt","dxu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.nx+4).astype(np.float))

        for a in ("dyt","dyu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.ny+4).astype(np.float))

        for a in ("cosu","cost"):
            self.set_attribute(a,2*np.random.rand(self.ny+4)-1.)

        for a in ("dzt","dzw","zw"):
            self.set_attribute(a,100*np.random.rand(self.nz))

        for a in ("eke_topo_hrms","eke_topo_lam","hrms_k0","coriolis_t","beta","eke_lee_flux","eke_bot_flux","L_Rossby"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4))

        for a in ("eke_len","K_diss_h","K_diss_gm","P_diss_skew","P_diss_hmix","P_diss_iso",
                  "kappaM","eke_diss_iw","eke_diss_tke","K_gm","flux_east","flux_north","flux_top","L_Rhines"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("eke","deke","Nsqr","u","v",):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a,np.random.randint(0,2,size=(self.nx+4,self.ny+4,self.nz)).astype(np.float))

        self.set_attribute("kbot",np.random.randint(0, self.nz, size=(self.nx+4,self.ny+4)))

        self.test_module = eke
        pyom_args = (self.pyom_new,)
        pyom_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["init_eke"] = (pyom_args, pyom_legacy_args)
        self.test_routines["set_eke_diffusivities"] = (pyom_args, pyom_legacy_args)
        self.test_routines["integrate_eke"] = (pyom_args, pyom_legacy_args)

    def test_passed(self,routine):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","eke","deke","hrms_k0","L_Rossby",
                  "L_Rhines","eke_len","K_gm","kappa_gm","K_iso","sqrteke","c_lee","c_Ri_diss",
                  "eke_diss_iw","eke_diss_tke","eke_lee_flux","eke_bot_flux"):
            passed = self._check_var(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

    def _check_var(self,var):
        v1, v2 = self.get_attribute(var)
        if v1 is None or v2 is None:
            raise RuntimeError(var)
        passed = np.allclose(v1[2:-2, 2:-2, ...], v2[2:-2, 2:-2, ...])
        if not passed:
            print(var, np.abs(v1[2:-2, 2:-2, ...]-v2[2:-2, 2:-2, ...]).max(), v1.max(), v2.max(), np.where(v1 != v2))
            while v1.ndim > 2:
                v1 = v1[...,-1]
            while v2.ndim > 2:
                v2 = v2[...,-1]
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
    test = EKETest(80, 70, 50, fortran=sys.argv[1])
    passed = test.run()
