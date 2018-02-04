from collections import OrderedDict
import numpy as np

from test_base import VerosUnitTest
from veros.core import isoneutral


class IsoneutralTest(VerosUnitTest):
    nx, ny, nz = 70, 60, 50
    extra_settings = {
        "enable_neutral_diffusion": True,
        "enable_skew_diffusion": True,
        "enable_TEM_friction": True,
    }
    test_module = isoneutral

    def initialize(self):
        m = self.veros_legacy.main_module

        for a in ("iso_slopec", "iso_dslope", "K_iso_steep", "dt_tracer", "dt_mom"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt", "dxu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.nx + 4).astype(np.float))

        for a in ("dyt", "dyu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.ny + 4).astype(np.float))

        for a in ("cosu", "cost"):
            self.set_attribute(a, 2 * np.random.rand(self.ny + 4) - 1.)

        for a in ("zt", "dzt", "dzw"):
            self.set_attribute(a, 100 * np.random.rand(self.nz))

        for a in ("area_u", "area_v", "area_t"):
            self.set_attribute(a, 1e5 * np.random.rand(self.nx + 4, self.ny + 4))

        for a in ("flux_east", "flux_north", "flux_top", "u_wgrid", "v_wgrid", "w_wgrid",
                  "K_iso", "K_gm", "kappa_gm", "du_mix", "P_diss_iso", "P_diss_skew"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz))

        for a in ("salt", "temp", "int_drhodT", "int_drhodS"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz, 3))

        for a in ("maskU", "maskV", "maskW", "maskT"):
            self.set_attribute(a, np.random.randint(0, 2, size=(self.nx + 4, self.ny + 4, self.nz)).astype(np.float))

        self.set_attribute("kbot", np.random.randint(0, self.nz, size=(self.nx + 4, self.ny + 4)))

        istemp = bool(np.random.randint(0, 2))

        veros_args = (self.veros_new, self.veros_new.temp, istemp)
        veros_legacy_args = dict(is_=-1, ie_=m.nx + 2, js_=-1, je_=m.ny + 2, nz_=m.nz, tr=m.temp, istemp=istemp)

        self.test_routines = OrderedDict()
        self.test_routines["isoneutral_diffusion_pre"] = ((self.veros_new, ), dict())
        self.test_routines["isoneutral_diag_streamfunction"] = ((self.veros_new, ), dict())
        self.test_routines["isoneutral_diffusion"] = (veros_args, veros_legacy_args)
        self.test_routines["isoneutral_skew_diffusion"] = (veros_args, veros_legacy_args)
        self.test_routines["isoneutral_friction"] = ((self.veros_new, ), dict())
        # unused in PyOM
        #self.test_routines["isoneutral_diffusion_all"] = (veros_args, veros_legacy_args)

    def test_passed(self, routine):
        if routine == "isoneutral_diffusion_pre":
            for v in ("K_11", "K_22", "K_33", "Ai_ez", "Ai_nz", "Ai_bx", "Ai_by"):
                self.check_variable(v)
        elif routine == "isoneutral_diag_streamfunction":
            for v in ("B1_gm", "B2_gm"):
                self.check_variable(v)
        elif routine == "isoneutral_friction":
            for v in ("K_diss_gm", "u", "du_mix", "v", "dv_mix", "flux_top"):
                self.check_variable(v)
        else:
            for f in ("flux_east", "flux_north", "flux_top", "dtemp_iso", "dsalt_iso", "temp", "salt", "P_diss_iso"):
                self.check_variable(f)


def test_isoneutral():
    IsoneutralTest().run()
