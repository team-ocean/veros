from test_base import VerosPyOMSystemTest
from acc2_test import ACC2


class ACC2NoEnergyConservationTest(VerosPyOMSystemTest):
    Testclass = ACC2
    timesteps = 5
    extra_settings = {'enable_conserve_energy': False}

    def test_passed(self):
        differing_scalars = self.check_scalar_objects()
        differing_arrays = self.check_array_objects()

        if differing_scalars or differing_arrays:
            print('The following attributes do not match between old and new veros:')
            for s, (v1, v2) in differing_scalars.items():
                print('{}, {}, {}'.format(s, v1, v2))
            for a, (v1, v2) in differing_arrays.items():
                if 'salt' in a or a in ('B1_gm', 'B2_gm'): # salt and isoneutral streamfunctions aren't used by this example
                    continue
                self.check_variable(a, atol=1e-6, data=(v1, v2))


def test_acc2_no_conservation(pyom2_lib, backend):
    ACC2NoEnergyConservationTest(fortran=pyom2_lib, backend=backend).run()
