from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom import momentum, external

class MomentumTest(PyOMTest):
    extra_settings = {
                        "coord_degree": True,
                        "enable_cyclic_x": True,
                        "enable_hydrostatic": True,
                        "enable_conserve_energy": True,
                        "enable_bottom_friction_var": True,
                        "enable_hor_friction_cos_scaling": True,
                        "enable_implicit_vert_friction": True,
                        "enable_explicit_vert_friction": True,
                        "enable_TEM_friction": True,
                        "enable_hor_friction": True,
                        "enable_biharmonic_friction": True,
                        "enable_ray_friction": True,
                        "enable_bottom_friction": True,
                        "enable_quadratic_bottom_friction": True,
                        "enable_momentum_sources": True,
                        "enable_streamfunction": True,
                        "congr_epsilon": 1e-12,
                     }
    first = True
    def initialize(self):
        m = self.pyom_legacy.main_module

        np.random.seed(123456)
        self.set_attribute("hor_friction_cosPower", np.random.randint(1,5))

        for a in ("dt_mom", "r_bot", "r_quad_bot", "A_h", "A_hbi", "AB_eps"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt","dxu"):
            self.set_attribute(a,np.random.randint(1,100,size=self.nx+4).astype(np.float))

        for a in ("dyt","dyu","tantr"):
            self.set_attribute(a,np.random.randint(1,100,size=self.ny+4).astype(np.float))

        for a in ("cosu","cost"):
            self.set_attribute(a,2*np.random.rand(self.ny+4)-1.)

        for a in ("dzt","dzw","zw"):
            self.set_attribute(a,np.random.rand(self.nz))

        for a in ("r_bot_var_u","r_bot_var_v", "surface_taux", "surface_tauy", "coriolis_t","coriolis_h"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4))

        for a in ("area_u","area_v","area_t", "hur", "hvr"):
            self.set_attribute(a,np.random.rand(self.nx+4,self.ny+4))

        for a in ("K_diss_v", "kappaM", "flux_north", "flux_east", "flux_top", "K_diss_bot", "K_diss_h",
                  "du_mix", "dv_mix", "dw_mix", "u_source", "v_source", "du_adv", "dv_adv"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("u","v","w","du","dv","dw"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a,np.random.randint(0,2,size=(self.nx+4,self.ny+4,self.nz)).astype(np.float))

        self.set_attribute("kbot",np.random.randint(0, self.nz) * np.ones((self.nx+4,self.ny+4), dtype=np.int))

        if self.first:
            external.streamfunction_init(self.pyom_new)
            self.pyom_legacy.fortran.streamfunction_init()
            self.first = False

        self.test_module = momentum
        pyom_args = (self.pyom_new,)
        pyom_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["momentum_advection"] = (pyom_args, pyom_legacy_args)
        self.test_routines["vertical_velocity"] = (pyom_args, pyom_legacy_args)
        self.test_routines["momentum"] = (pyom_args, pyom_legacy_args)


    def test_passed(self,routine):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","u","v","w","K_diss_v","du_adv","dv_adv","du","dv","dw",
                  "K_diss_bot","K_diss_h","du_mix","dv_mix","psi","dpsi","du_cor","dv_cor"):
            passed = self._check_var(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

    def _check_var(self,var):
        v1, v2 = self.get_attribute(var)
        passed = np.allclose(v1, v2)
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
    test = MomentumTest(80, 70, 50, fortran=sys.argv[1])
    passed = test.run()
