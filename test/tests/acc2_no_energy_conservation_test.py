import sys

from test_base import VerosRunTest
from acc2_test import ACC2

class ACC2NoEngConvTest(VerosRunTest):
    Testclass = ACC2
    timesteps = 5
    extra_settings = {"enable_conserve_energy": False}

if __name__ == "__main__":
    test = ACC2NoEngConvTest()
    passed = test.run()
    sys.exit(int(not passed))
