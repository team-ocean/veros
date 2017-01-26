import imp

from climate.pyom import PyOM, diagnostics

class LowercaseAttributeWrapper(object):
    """
    A simple wrapper class that converts attributes to lower case (needed for Fortran interface)
    """
    def __init__(self,wrapped_object):
        object.__setattr__(self,"_w",wrapped_object)

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self,"_w"),key.lower())

    def __setattr__(self, key, value):
        setattr(self._w,key.lower(),value)

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
            self.fortran = LowercaseAttributeWrapper(imp.load_dynamic("pyOM_code", fortran))
            self.main_module = LowercaseAttributeWrapper(self.fortran.main_module)
            self.isoneutral_module = LowercaseAttributeWrapper(self.fortran.isoneutral_module)
            self.idemix_module = LowercaseAttributeWrapper(self.fortran.idemix_module)
            self.tke_module = LowercaseAttributeWrapper(self.fortran.tke_module)
            self.eke_module = LowercaseAttributeWrapper(self.fortran.eke_module)
        else:
            self.legacy_mode = False
            self.fortran = self
            self.main_module = self
            self.isoneutral_module = self
            self.idemix_module = self
            self.tke_module = self
            self.eke_module = self

        super(PyOMLegacy,self).__init__(*args, **kwargs)

    def set_legacy_parameter(self, *args, **kwargs):
        m = self.fortran.main_module
        self.onx = 2
        self.is_pe = 2
        self.ie_pe = m.nx+2
        self.js_pe = 2
        self.je_pe = m.ny+2
        self.if2py = lambda i: i+self.onx-self.is_pe
        self.jf2py = lambda j: j+self.onx-self.js_pe
        self.ip2fy = lambda i: i+self.is_pe-self.onx
        self.jp2fy = lambda j: j+self.js_pe-self.onx

    def setup(self, *args, **kwargs):
        if self.legacy_mode:
            self.fortran.my_mpi_init(0)
            self.set_parameter()
            self.set_legacy_parameter()
            self.fortran.pe_decomposition()
            self.fortran.allocate_main_module()
            self.fortran.allocate_isoneutral_module()
            self.fortran.allocate_tke_module()
            self.fortran.allocate_eke_module()
            self.fortran.allocate_idemix_module()
            self.set_grid()
            self.fortran.calc_grid()
            self.set_coriolis()
            self.fortran.calc_beta()
            self.set_topography()
            self.fortran.calc_topo()
            self.fortran.calc_spectral_topo()
            self.set_initial_conditions()
            self.fortran.calc_initial_conditions()
            self.set_forcing()
            if self.fortran.main_module.enable_streamfunction:
                self.fortran.streamfunction_init()
            self.fortran.check_isoneutral_slope_crit()
        else:
            # self.set_parameter() is called twice, but that shouldn't matter
            self.set_parameter()
            self.set_legacy_parameter()
            super(PyOMLegacy,self).setup(*args,**kwargs)
