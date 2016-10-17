import climate
from climate.data import dist
import unittest


class MonteCarlo(unittest.TestCase):
    samples = 10
    num = 0
    inRange = True
    intMin = 5
    intMax = 10
    floatMin = 5.
    floatMax = 10.

    def add1(self, a):
        self.num += 1

    def isInRangeInt(self, num):
        if num < self.intMin or self.intMax <= num:
            self.inRange = False

    def isInRangeFloat(self, num):
        if num < self.floatMin or self.floatMax <= num:
            self.inRange = False

    def test_correctcalls_randomint(self):
        climate.data.montecarlo(self.add1, self.samples, a = (5, 10, dist.random))
        self.assertTrue(self.num == self.samples)

    def test_correctcalls_randomfloat(self):
        climate.data.montecarlo(self.add1, self.samples, a = (5., 10., dist.random))
        self.assertTrue(self.num == self.samples)

    def test_correctcalls_normal(self):
        climate.data.montecarlo(self.add1, self.samples, a = (5, 10, dist.normal))
        self.assertTrue(self.num == self.samples)

    def test_correctcalls_constant(self):
        climate.data.montecarlo(self.add1, self.samples, a = 5)
        self.assertTrue(self.num == self.samples)

    def test_randomint(self):
        climate.data.montecarlo(self.isInRangeInt, 100, num = (self.intMin, self.intMax, dist.random))
        self.assertTrue(self.inRange)

    def test_randomfloat(self):
        climate.data.montecarlo(self.isInRangeFloat, 100, num = (self.floatMin, self.floatMax, dist.random))
        self.assertTrue(self.inRange)

if __name__ == '__main__':
    unittest.main()
