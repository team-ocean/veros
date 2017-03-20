import unittest

loader = unittest.TestLoader()
unittest.TextTestRunner().run(loader.discover("."))
