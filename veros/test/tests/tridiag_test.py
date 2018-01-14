from builtins import range

import numpy as np
import bohrium as bh
import sys

from test_base import VerosUnitTest
from veros.core import numerics


class TridiagTest(VerosUnitTest):
    nx, ny, nz = 100, 100, 100


    def initialize(self):
        pass


    def run(self):
        a, b, c, d = (np.random.randn(self.nx, self.ny, self.nz) for _ in range(4))

        out_legacy = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nx):
            for j in range(self.ny):
                out_legacy[i, j] = self.veros_legacy.fortran.solve_tridiag(
                    a=a[i,j], b=b[i,j], c=c[i,j], d=d[i,j], n=self.nz)

        if self.veros_new.backend_name == "bohrium":
            a, b, c, d = (bh.array(v) for v in (a, b, c, d))

        out_new = numerics.solve_tridiag(self.veros_new, a, b, c, d)
        passed = np.allclose(out_legacy, out_new)

        return passed


if __name__ == "__main__":
    passed = TridiagTest().run()
    sys.exit(int(not passed))
