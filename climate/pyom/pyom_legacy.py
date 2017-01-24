import imp

from climate.pyom import PyOM, diagnostics

class PyOMLegacy(PyOM):
    """
    PyOM class that supports the PyOM Fortran interface

    """
    def __init__(self, fortran=None, *args, **kwargs):
        """
        To use the pyom2 legacy interface point the fortran argument to the PyOM fortran library:

        > simulation = GlobalOneDegree(fortran = "pyOM_code.so")

        """
        if fortran:
            self.legacy_mode = True
            self.fortran = imp.load_source("pyom_legacy", fortran)
        else:
            self.legacy_mode = False
            self.fortran = self
            self.main_module = self
            self.isoneutral_module = self
            self.idemix_module = self
            self.tke_module = self
            self.eke_module = self

        super(PyOMLegacy,self).__init__(*args, **kwargs)

        self.register_average = lambda *args, **kwargs: diagnostics.register_average(*args,pyom=self,**kwargs)
        self.__getattr__ = lambda self, attr: getattr(self, attr.lower() if self.legacy_mode else attr)
        self.__setattr__ = lambda self, attr, val: setattr(self, attr.lower() if self.legacy_mode else attr, val)

    def set_parameter(self, *args, **kwargs):
        super(PyOMLegacy,self).set_parameter(*args,**kwargs)
        self.onx = 2
        self.is_pe = 2
        self.ie_pe = self.nx+2
        self.js_pe = 2
        self.je_pe = self.ny+2
        self.if2py = lambda i: i+self.onx-self.is_pe
        self.jf2py = lambda j: j+self.onx-self.js_pe
        self.ip2fy = lambda i: i+self.is_pe-self.onx
        self.jp2fy = lambda j: j+self.js_pe-self.onx
