from climate.setup.acc2 import ACC2
from pyomtest import PyOMTest

import sys
import numpy as np
import matplotlib.pyplot as plt

class ACC2Test(PyOMTest):
    timesteps = 1
    def __init__(self, fortran):
        self.fortran = fortran

    def _check_all_objects(self):
        differing_scalars = self.check_scalar_objects()
        differing_arrays = self.check_array_objects()
        if differing_scalars or differing_arrays:
            print("The following attributes do not match between old and new pyom:")
            for s, (v1, v2) in differing_scalars.items():
                print("{}, {}, {}".format(s,v1,v2))
            for a, (v1, v2) in differing_arrays.items():
                if v1 is None:
                    print(a, v1, "")
                    continue
                if v2 is None:
                    print(a, "", v2)
                    continue
                self._check_var(a)
        plt.show()

    def _normalize(self,*arrays):
        norm = np.abs(arrays[0]).max()
        if norm == 0.:
            return arrays
        return (a / norm for a in arrays)

    def _check_var(self,var):
        if "salt" in var or var in ("B1_gm","B2_gm"): # salt and isoneutral streamfunctions aren't used by this example
            return True
        v1, v2 = self.get_attribute(var)
        if v1.ndim > 1:
            v1 = v1[2:-2, 2:-2, ...]
        if v2.ndim > 1:
            v2 = v2[2:-2, 2:-2, ...]
        if v1 is None or v2 is None:
            raise RuntimeError(var)
        try:
            passed = np.allclose(*self._normalize(v1,v2),atol=1e-6)
        except ValueError:
            print(var)
            raise
        if not passed:
            print(var, np.abs(v1-v2).max(), v1.max(), v2.max(), np.where(v1 != v2))
            while v1.ndim > 2:
                v1 = v1[...,-1]
            while v2.ndim > 2:
                v2 = v2[...,-1]
            if v1.ndim == 2:
                fig, axes = plt.subplots(1,3)
                axes[0].imshow(v1)
                axes[0].set_title("New")
                axes[1].imshow(v2)
                axes[1].set_title("Legacy")
                axes[2].imshow(v1 - v2)
                axes[2].set_title("diff")
                fig.suptitle(var)
        return passed

    def run(self):
        self.pyom_new = ACC2()
        self.pyom_legacy = ACC2(fortran=self.fortran)
        # integrate for some time steps and compare
        if self.timesteps == 0:
            self.pyom_new.setup()
            self.pyom_legacy.setup()
        else:
            self.pyom_new.run(runlen = self.timesteps * 86400. / 2, snapint=1e10)
            self.pyom_legacy.run(runlen = self.timesteps * 86400. / 2, snapint=1e10)
        self._check_all_objects()

if __name__ == "__main__":
    test = ACC2Test(sys.argv[1])
    test.run()
