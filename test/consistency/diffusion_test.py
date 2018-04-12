from collections import OrderedDict

import pytest

import numpy as np

from test_base import VerosPyOMUnitTest
from veros.core import diffusion, numerics


class DiffusionTest(VerosPyOMUnitTest):
    nx, ny, nz = 70, 60, 50
    extra_settings = {
        "enable_cyclic_x": True,
        "enable_conserve_energy": True,
        "enable_hor_friction_cos_scaling": True,
        "enable_tempsalt_sources": True,
    }

    def initialize(self):
        self.set_attribute("hor_friction_cosPower", np.random.randint(1, 5))

        for a in ("dt_tracer", "K_hbi", "K_h"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt", "dxu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.nx + 4).astype(np.float))

        for a in ("dyt", "dyu"):
            self.set_attribute(a, np.random.randint(1, 100, size=self.ny + 4).astype(np.float))

        for a in ("cosu", "cost"):
            self.set_attribute(a, 2 * np.random.rand(self.ny + 4) - 1.)

        for a in ("dzt", "dzw"):
            self.set_attribute(a, 100 * np.random.rand(self.nz))

        for a in ("flux_east", "flux_north", "flux_top", "dtemp_hmix", "dsalt_hmix",
                  "temp_source", "salt_source"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz))

        for a in ("temp", "salt", "int_drhodS", "int_drhodT"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz, 3))

        self.set_attribute("kbot", np.random.randint(0, self.nz, size=(self.nx + 4, self.ny + 4)))
        numerics.calc_topo(self.veros_new)
        self.veros_legacy.call_fortran_routine("calc_topo")

        self.set_attribute("P_diss_hmix", np.random.randn(self.nx + 4, self.ny + 4, self.nz) * self.veros_new.maskT)
        self.test_module = diffusion
        veros_args = (self.veros_new, )
        veros_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines.update(
            tempsalt_biharmonic=(veros_args, veros_legacy_args),
            tempsalt_diffusion=(veros_args, veros_legacy_args),
            tempsalt_sources=(veros_args, veros_legacy_args),
        )

    def test_passed(self, routine):
        for f in ("flux_east", "flux_north", "flux_top", "temp", "salt", "P_diss_hmix",
                  "dtemp_hmix", "dsalt_hmix", "P_diss_sources"):
            self.check_variable(f)


@pytest.mark.pyom
def test_diffusion(pyom2_lib, backend):
    DiffusionTest(fortran=pyom2_lib, backend=backend).run()
