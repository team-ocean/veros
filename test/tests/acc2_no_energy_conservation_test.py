import sys

from test_base import VerosRunTest
from acc2_test import ACC2

class ACC2NoEngConvTest(VerosRunTest):
    Testclass = ACC2
    timesteps = 5
    extra_settings = {"enable_conserve_energy": False}

    def test_passed(self):
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
                    v1[2,:] = 0.
                    v2[2,:] = 0.
                passed = self.check_variable(a,atol=1e-5,data=(v1,v2)) and passed
        # plt.show()
        return passed

if __name__ == "__main__":
    test = ACC2NoEngConvTest()
    passed = test.run()
    sys.exit(int(not passed))
