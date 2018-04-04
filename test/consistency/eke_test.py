from collections import OrderedDict

import pytest

import numpy as np

from test_base import VerosPyOMUnitTest
from veros.core import eke


class EKETest(VerosPyOMUnitTest):
    nx, ny, nz = 70, 60, 50
    extra_settings = {
        "enable_cyclic_x": True,
        "enable_eke_leewave_dissipation": True,
        "enable_eke": True,
        "enable_TEM_friction": True,
        "enable_eke_isopycnal_diffusion": True,
        "enable_store_cabbeling_heat": True,
        "enable_eke_superbee_advection": True,
        "enable_eke_upwind_advection": True
    }

    def initialize(self):
        for a in ("eke_hrms_k0_min", "eke_k_max", "eke_c_k", "eke_crhin", "eke_cross",
                  "eke_lmin", "K_gm_0", "K_iso_0", "c_lee0", "eke_Ri0", "eke_Ri1", "eke_int_diss0",
                  "kappa_EKE0", "eke_r_bot", "eke_c_eps", "alpha_eke", "dt_tracer", "AB_eps"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt", "dxu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.nx + 4).astype(np.float))

        for a in ("dyt", "dyu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.ny + 4).astype(np.float))

        for a in ("cosu", "cost"):
            self.set_attribute(a, 2 * np.random.rand(self.ny + 4) - 1.)

        for a in ("dzt", "dzw", "zw"):
            self.set_attribute(a, 100 * np.random.rand(self.nz))

        for a in ("eke_topo_hrms", "eke_topo_lam", "hrms_k0", "coriolis_t", "beta",
                  "eke_lee_flux", "eke_bot_flux", "L_rossby"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4))

        for a in ("eke_len", "K_diss_h", "K_diss_gm", "P_diss_skew", "P_diss_hmix", "P_diss_iso",
                  "kappaM", "eke_diss_iw", "eke_diss_tke", "K_gm", "flux_east", "flux_north", "flux_top",
                  "L_rhines"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz))

        for a in ("eke", "deke", "Nsqr", "u", "v"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz, 3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a, np.random.randint(0, 2, size=(self.nx + 4, self.ny + 4, self.nz)).astype(np.float))

        kbot = np.random.randint(1, self.nz, size=(self.nx + 4, self.ny + 4))
        # add some islands, but avoid boundaries
        kbot[3:-3, 3:-3].flat[np.random.randint(0, (self.nx - 2) * (self.ny - 2), size=10)] = 0
        self.set_attribute("kbot", kbot)

        self.test_module = eke
        veros_args = (self.veros_new, )
        veros_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["init_eke"] = (veros_args, veros_legacy_args)
        self.test_routines["set_eke_diffusivities"] = (veros_args, veros_legacy_args)
        self.test_routines["integrate_eke"] = (veros_args, veros_legacy_args)

    def test_passed(self, routine):
        for f in ("flux_east", "flux_north", "flux_top", "eke", "deke", "hrms_k0", "L_rossby",
                  "L_rhines", "eke_len", "K_gm", "kappa_gm", "K_iso", "sqrteke", "c_lee", "c_Ri_diss",
                  "eke_diss_iw", "eke_diss_tke", "eke_lee_flux", "eke_bot_flux"):
            self.check_variable(f)


@pytest.mark.pyom
def test_eke(pyom2_lib, backend):
    EKETest(fortran=pyom2_lib, backend=backend).run()
