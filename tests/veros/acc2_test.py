from veros.setup.acc2 import ACC2
from test_base import VerosTest

import sys
import numpy as np
import matplotlib.pyplot as plt

class ACC2Test(VerosTest):
    timesteps = 5

    def __init__(self, *args, **kwargs):
        try:
            self.fortran = kwargs["fortran"]
        except KeyError:
            try:
                self.fortran = sys.argv[1]
            except IndexError:
                raise RuntimeError("Path to fortran library must be given via keyword argument or command line")

    def _check_all_objects(self):
        differing_scalars = self.check_scalar_objects()
        differing_arrays = self.check_array_objects()
        passed = True
        if differing_scalars or differing_arrays:
            print("The following attributes do not match between old and new veros:")
            for s, (v1, v2) in differing_scalars.items():
                print("{}, {}, {}".format(s,v1,v2))
            for a, (v1, v2) in differing_arrays.items():
                if v1 is None:
                    print(a, v1, "")
                    continue
                if v2 is None:
                    print(a, "", v2)
                    continue
                if "salt" in a or a in ("B1_gm","B2_gm"): # salt and isoneutral streamfunctions aren't used by this example
                    continue
                if a in ("psi", "psin"): # top row contains noise and is not part of the solution
                    v1[1:,:] = 0.
                    v2[1:,:] = 0.
                passed = self.check_variable(a,atol=1e-5) and passed
        plt.show()
        return passed

    def run(self):
        self.veros_new = ACC2()
        self.veros_new.pyom_compatibility_mode = True
        self.veros_legacy = ACC2(fortran=self.fortran)
        # integrate for some time steps and compare
        if self.timesteps == 0:
            self.veros_new.setup()
            self.veros_legacy.setup()
        else:
            self.veros_new.run(runlen = self.timesteps * 86400. / 2)
            self.veros_legacy.run(runlen = self.timesteps * 86400. / 2, snapint=1e10)
        return self._check_all_objects()

if __name__ == "__main__":
    passed = ACC2Test().run()
    sys.exit(int(not passed))
