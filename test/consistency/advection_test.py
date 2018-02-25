from collections import OrderedDict

import pytest

import numpy as np

from test_base import VerosPyOMUnitTest
from veros.core import advection, numerics


class AdvectionTest(VerosPyOMUnitTest):
    nx, ny, nz = 70, 60, 50

    def initialize(self):
        self.set_attribute("dt_tracer", 3600.)

        for a in ("dxt", ):
            self.set_attribute(a, np.random.randint(1, 100, size=self.nx + 4).astype(np.float))

        for a in ("dyt", ):
            self.set_attribute(a, np.random.randint(1, 100, size=self.ny + 4).astype(np.float))

        for a in ("cosu", "cost"):
            self.set_attribute(a, 2 * np.random.rand(self.ny + 4) - 1.)

        for a in ("dzt", "dzw"):
            self.set_attribute(a, 100 * np.random.rand(self.nz))

        for a in ("flux_east", "flux_north", "flux_top", "u_wgrid", "v_wgrid", "w_wgrid"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz))

        for a in ("u", "v", "w", "Hd"):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz, 3))

        self.set_attribute("kbot", np.random.randint(0, self.nz, size=(self.nx + 4, self.ny + 4)).astype(np.float))

        numerics.calc_topo(self.veros_new)
        self.veros_legacy.call_fortran_routine("calc_topo")

        self.test_module = advection
        veros_args = (self.veros_new, self.veros_new.flux_east, self.veros_new.flux_north,
                      self.veros_new.flux_top, self.veros_new.Hd[..., 1])
        veros_legacy_args = dict(
            is_=-1, ie_=self.nx + 2, js_=-1, je_=self.ny + 2, nz_=self.nz,
            adv_fe=self.veros_legacy.get_fortran_attribute("flux_east"),
            adv_fn=self.veros_legacy.get_fortran_attribute("flux_north"),
            adv_ft=self.veros_legacy.get_fortran_attribute("flux_top"),
            var=self.veros_legacy.get_fortran_attribute("Hd")[..., 1]
        )
        self.test_routines = OrderedDict()
        self.test_routines["calculate_velocity_on_wgrid"] = ((self.veros_new, ), dict())
        self.test_routines.update(
            adv_flux_2nd=(veros_args, veros_legacy_args),
            adv_flux_superbee=(veros_args, veros_legacy_args),
            adv_flux_upwind_wgrid=(veros_args, veros_legacy_args),
            adv_flux_superbee_wgrid=(veros_args, veros_legacy_args)
        )

    def test_passed(self, routine):
        if routine == "calculate_velocity_on_wgrid":
            for v in ("u_wgrid", "v_wgrid", "w_wgrid"):
                self.check_variable(v)
        else:
            for f in ("flux_east", "flux_north", "flux_top"):
                self.check_variable(f)


@pytest.mark.pyom
def test_advection(pyom2_lib, backend):
    AdvectionTest(fortran=pyom2_lib, backend=backend).run()
