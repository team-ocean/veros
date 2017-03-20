import numpy as np
import sys

from test_base import PyOMTest
from climate.pyom.core import numerics

class TridiagTest(PyOMTest):
    def initialize(self):
        pass

    def run(self):
        for _ in range(100):
            a, b, c, d = (np.random.randn(self.nz) for _ in range(4))
            out_legacy = self.pyom_legacy.fortran.solve_tridiag(a=a,b=b,c=c,d=d,n=self.nz)
            out_new = numerics.solve_tridiag(self.pyom_new,a,b,c,d)
            if not np.allclose(out_legacy, out_new):
                return False
            return True

if __name__ == "__main__":
    test = TridiagTest(100, 100, 200, fortran=sys.argv[1])
    passed = test.run()
    sys.exit(int(not passed))
