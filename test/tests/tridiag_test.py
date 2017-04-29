import numpy as np
import sys

from test_base import VerosTest
from veros.core import numerics

class TridiagTest(VerosTest):
    nx, ny, nz = 100, 100, 200
    def initialize(self):
        pass

    def run(self):
        for _ in range(100):
            a, b, c, d = (np.random.randn(self.nz) for _ in range(4))
            out_legacy = self.veros_legacy.fortran.solve_tridiag(a=a,b=b,c=c,d=d,n=self.nz)
            out_new = numerics.solve_tridiag(self.veros_new,a,b,c,d)
            if not np.allclose(out_legacy, out_new):
                return False
            return True

if __name__ == "__main__":
    passed = TridiagTest().run()
    sys.exit(int(not passed))
