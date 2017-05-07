from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from test_base import VerosTest
from veros.core import momentum, external, numerics

class MomentumTest(VerosTest):
    nx, ny, nz = 70, 60, 50
    extra_settings = {
                        "coord_degree": True,
                        "enable_cyclic_x": True,
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
                        "congr_epsilon": 1e-12,
                        "congr_max_iterations": 10000,
                     }
    first = True
    def initialize(self):
        m = self.veros_legacy.main_module

        np.random.seed(123456)
        self.set_attribute("hor_friction_cosPower", np.random.randint(1,5))

        for a in ("dt_mom", "r_bot", "r_quad_bot", "A_h", "A_hbi", "AB_eps", "x_origin", "y_origin"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt",):
            self.set_attribute(a, 0.1 * np.ones(self.nx+4) + 0.01 * np.random.rand(self.nx+4))

        for a in ("dyt",):
            self.set_attribute(a, 0.1 * np.ones(self.ny+4) + 0.01 * np.random.rand(self.ny+4))

        for a in ("dzt",):
            self.set_attribute(a,np.random.rand(self.nz))

        for a in ("r_bot_var_u", "r_bot_var_v", "surface_taux", "surface_tauy", "coriolis_t", "coriolis_h"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4))

        for a in ("K_diss_v", "kappaM", "flux_north", "flux_east", "flux_top", "K_diss_bot", "K_diss_h",
                  "du_mix", "dv_mix", "u_source", "v_source", "du_adv", "dv_adv"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("u","v","w","du","dv"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        kbot = np.random.randint(1, self.nz, size=(self.nx+4,self.ny+4))
        # add some islands, but avoid boundaries
        kbot[3:-3,3:-3].flat[np.random.randint(0, (self.nx-2) * (self.ny-2), size=10)] = 0
        self.set_attribute("kbot",kbot)

        numerics.calc_grid(self.veros_new)
        numerics.calc_topo(self.veros_new)
        self.veros_legacy.fortran.calc_grid()
        self.veros_legacy.fortran.calc_topo()

        if self.first:
            external.streamfunction_init(self.veros_new)
            self.veros_legacy.fortran.streamfunction_init()
            self.first = False

        self.test_module = momentum
        veros_args = (self.veros_new,)
        veros_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["momentum_advection"] = (veros_args, veros_legacy_args)
        self.test_routines["vertical_velocity"] = (veros_args, veros_legacy_args)
        self.test_routines["momentum"] = (veros_args, veros_legacy_args)


    def test_passed(self,routine):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","u","v","w","K_diss_v","du_adv","dv_adv","du","dv",
                  "K_diss_bot","K_diss_h","du_mix","dv_mix","psi","dpsi","du_cor","dv_cor"):
            passed = self.check_variable(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

if __name__ == "__main__":
    passed = MomentumTest().run()
    sys.exit(int(not passed))
