from collections import OrderedDict

import pytest

import numpy as np

from test_base import VerosPyOMUnitTest
from veros.core import idemix


class IdemixTest(VerosPyOMUnitTest):
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
        for a in ("gamma", "mu0", "jstar", "eke_diss_surfbot_frac", "dt_tracer", "tau_v", "AB_eps"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt", "dxu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.nx + 4).astype(np.float))

        for a in ("dyt", "dyu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.ny + 4).astype(np.float))

        for a in ("cosu", "cost"):
            self.set_attribute(a, 2 * np.random.rand(self.ny + 4) - 1.)

        for a in ("zt", "dzt", "dzw"):
            self.set_attribute(a, np.random.rand(self.nz))

        for a in ("area_u", "area_v", "area_t", "coriolis_t", "forc_iw_bottom", "forc_iw_surface"):
            self.set_attribute(a, np.random.rand(self.nx + 4, self.ny + 4))

        for a in ("c0", "v0", "alpha_c", "eke_diss_iw", "K_diss_gm", "K_diss_h", "K_iso", "K_gm",
                  "kappa_gm", "P_diss_iso", "P_diss_skew", "P_diss_hmix", "K_diss_bot"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz))

        for a in ("Nsqr", "E_iw", "dE_iw"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz, 3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a, np.random.randint(0, 2, size=(self.nx + 4, self.ny + 4, self.nz)).astype(np.float))

        self.set_attribute("kbot", np.random.randint(0, self.nz, size=(self.nx + 4, self.ny + 4)))

        self.test_routines = OrderedDict()
        self.test_routines["set_idemix_parameter"] = ((self.veros_new, ), dict())
        self.test_routines["integrate_idemix"] = ((self.veros_new, ), dict())

    def test_passed(self, routine):
        if routine == "set_idemix_parameter":
            for v in ("c0", "v0", "alpha_c"):
                self.check_variable(v, atol=1e-7)
        elif routine == "integrate_idemix":
            for v in ("E_iw", "dE_iw", "iw_diss", "flux_east", "flux_north", "flux_top"):
                self.check_variable(v)


@pytest.mark.pyom
def test_idemix(pyom2_lib):
    IdemixTest(fortran=pyom2_lib).run()
