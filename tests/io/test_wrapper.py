import climate, numpy
import unittest

class IOWrapper(unittest.TestCase):
    def test_getdata(self):
        a = climate.io.wrapper(numpy.arange(5))
        self.assertTrue(len(a) == 5)
        self.assertIsInstance(a, climate.io.wrapper)
        for i in xrange(len(a)):
            self.assertTrue(a[i] == i)

    def test_setdata(self):
        a = climate.io.wrapper(numpy.zeros(5))
        self.assertTrue(len(a) == 5)
        for i in xrange(len(a)):
            a[i] = i
            self.assertTrue(a[i] == i)

    def test_specialindex(self):
        a = climate.io.wrapper(numpy.arange(5))
        a[a < 3] = 3
        self.assertIsInstance(a, climate.io.wrapper)
        for i in xrange(len(a)):
            if i < 3:
                self.assertTrue(a[i] == 3)
            else:
                self.assertTrue(a[i] == i)

    def test_add_minus(self):
        a = climate.io.wrapper(numpy.arange(5))
        b = a - a
        c = a + a

        self.assertIsInstance(a, climate.io.wrapper)
        self.assertIsInstance(b, climate.io.wrapper)
        self.assertIsInstance(c, climate.io.wrapper)
        for i in xrange(5):
            self.assertTrue(a[i] == i)
            self.assertTrue(b[i] == 0)
            self.assertTrue(c[i] == i*2)

    def test_comparison(self):
        a = climate.io.wrapper(numpy.arange(5))
        b = climate.io.wrapper(numpy.arange(1,6))
        a_eq = a == a
        ab_eq = a == b
        ab_lt = a < b
        ab_gt = b > a
        a_le = a <= a
        a_ge = a >= a
        for i in xrange(5):
            self.assertTrue(a_eq[i])
            self.assertFalse(ab_eq[i])
            self.assertTrue(ab_lt[i])
            self.assertTrue(ab_gt[i])
            self.assertTrue(a_le[i])
            self.assertTrue(a_ge[i])

    def test_rmath(self):
        a = climate.io.wrapper(numpy.arange(1,6))
        b = 5 / a
        c = 5 - a
        d = 5 + a
        e = 5 * a

        self.assertIsInstance(a, climate.io.wrapper)
        self.assertIsInstance(b, climate.io.wrapper)
        self.assertIsInstance(c, climate.io.wrapper)
        self.assertIsInstance(d, climate.io.wrapper)
        self.assertIsInstance(e, climate.io.wrapper)
        for i in xrange(5):
            j = i + 1
            self.assertTrue(b[i] == 5/j)
            self.assertTrue(c[i] == 5-j)
            self.assertTrue(d[i] == 5+j)
            self.assertTrue(e[i] == 5*j)

    def test_array_assign(self):
        a = climate.io.wrapper(numpy.arange(6))
        b = climate.io.wrapper(numpy.float(5.))
        a[3] = b
        self.assertIsInstance(a, climate.io.wrapper)
        self.assertTrue((a == numpy.array([0,1,2,5,4,5])).all())

    def test_notnumpy(self):
        a = climate.io.wrapper(5)
        b = a + 5
        self.assertTrue(b == 10)
        self.assertIsInstance(b, climate.io.wrapper)

    def test_imath(self):
        a = climate.io.wrapper(numpy.arange(5))
        b = a
        self.assertIsInstance(a, climate.io.wrapper)
        self.assertIsInstance(b, climate.io.wrapper)
        self.assertTrue((numpy.arange(5) == a).all())
        self.assertTrue((b == a).all())
        a += 5
        self.assertTrue((numpy.arange(5, 10) == a).all())
        self.assertIsInstance(a, climate.io.wrapper)
        a -= 5
        self.assertTrue((numpy.arange(5) == a).all())
        self.assertIsInstance(a, climate.io.wrapper)
        a *= 2
        self.assertTrue((numpy.arange(0, 10, 2) == a).all())
        self.assertIsInstance(a, climate.io.wrapper)
        a /= 2
        self.assertTrue((numpy.arange(5) == a).all())
        self.assertIsInstance(a, climate.io.wrapper)
        a += b
        self.assertTrue((numpy.arange(0,10,2) == a).all())
        self.assertIsInstance(a, climate.io.wrapper)

if __name__ == '__main__':
    unittest.main()
