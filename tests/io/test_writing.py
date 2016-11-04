import climate, numpy
import unittest
import os

class IOwrite(unittest.TestCase):
    def tearDown(self):
        if os.path.isfile("test"):
            os.remove("test")

    def test_writeNumpy(self):
        a = climate.io.wrapper(numpy.arange(50000))
        b = climate.io.wrapper(numpy.arange(50000))
        a.write("test")
        a += 2
        c = numpy.loadtxt("test", delimiter="\n")
        self.assertTrue((b == c).all())
