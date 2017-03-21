import sys
import os
import numpy as np
import bohrium as bh
from collections import OrderedDict

from climate.pyom import PyOMLegacy
from climate import Timer
import climate

flush = bh.flush

class PyOMTest(object):
    legacy_modules = ("main_module", "isoneutral_module", "tke_module",
                      "eke_module", "idemix_module")
    array_attribute_file = os.path.join(os.path.dirname(__file__), "array_attributes")
    scalar_attribute_file = os.path.join(os.path.dirname(__file__), "scalar_attributes")
    extra_settings = None
    test_module = None
    test_routines = None

    def __init__(self, dims=None, fortran=None):
        self.pyom_new = PyOMLegacy()
        if not fortran:
            try:
                fortran = sys.argv[1]
            except IndexError:
                raise RuntimeError("Path to fortran library must be given via keyword argument or command line")
        self.pyom_legacy = PyOMLegacy(fortran=fortran)

        if dims:
            self.nx, self.ny, self.nz = dims
        self.set_attribute("nx", self.nx)
        self.set_attribute("ny", self.ny)
        self.set_attribute("nz", self.nz)
        if self.extra_settings:
            for attribute, value in self.extra_settings.items():
                self.set_attribute(attribute, value)
        self.pyom_new.set_legacy_parameter()
        self.pyom_new._allocate()
        self.pyom_legacy.fortran.my_mpi_init(0)
        self.pyom_legacy.fortran.pe_decomposition()
        self.pyom_legacy.set_legacy_parameter()
        self.pyom_legacy.fortran.allocate_main_module()
        self.pyom_legacy.fortran.allocate_isoneutral_module()
        self.pyom_legacy.fortran.allocate_tke_module()
        self.pyom_legacy.fortran.allocate_eke_module()
        self.pyom_legacy.fortran.allocate_idemix_module()


    def set_attribute(self, attribute, value):
        if isinstance(value, np.ndarray):
            getattr(self.pyom_new, attribute)[...] = value
        else:
            setattr(self.pyom_new, attribute, value)
        for module in self.legacy_modules:
            module_handle = getattr(self.pyom_legacy,module)
            if hasattr(module_handle, attribute):
                try:
                    v = np.asfortranarray(value).copy2numpy()
                except AttributeError:
                    v = np.asfortranarray(value)
                setattr(module_handle, attribute, v)
                assert np.all(value == getattr(module_handle, attribute)), attribute
                return
        raise AttributeError("Legacy PyOM has no attribute {}".format(attribute))


    def get_attribute(self, attribute):
        try:
            pyom_attr = getattr(self.pyom_new, attribute)
        except AttributeError:
            pyom_attr = None
        try:
            pyom_attr = pyom_attr.copy2numpy()
        except AttributeError:
            pass
        pyom_legacy_attr = None
        for module in self.legacy_modules:
            module_handle = getattr(self.pyom_legacy,module)
            if hasattr(module_handle, attribute):
                pyom_legacy_attr = getattr(module_handle, attribute)
        return pyom_attr, pyom_legacy_attr


    def get_routine(self, routine, submodule=None):
        if submodule:
            pyom_module_handle = submodule
        else:
            pyom_module_handle = self.pyom_new
        pyom_routine = getattr(pyom_module_handle, routine)
        pyom_legacy_routine = getattr(self.pyom_legacy.fortran, routine)
        return pyom_routine, pyom_legacy_routine


    def get_all_attributes(self,attribute_file):
        attributes = {}
        with open(attribute_file,"r") as f:
            for a in f:
                a = a.strip()
                attributes[a] = self.get_attribute(a)
        return attributes


    def check_scalar_objects(self):
        differing_objects = {}
        scalars = self.get_all_attributes(self.scalar_attribute_file)
        for s, (v1,v2) in scalars.items():
            if ((v1 is None) != (v2 is None)) or v1 != v2:
                differing_objects[s] = (v1,v2)
        return differing_objects


    def check_array_objects(self):
        differing_objects = {}
        arrays = self.get_all_attributes(self.array_attribute_file)
        for a, (v1,v2) in arrays.items():
            if ((v1 is None) != (v2 is None)) or not np.array_equal(v1,v2):
                differing_objects[a] = (v1,v2)
        return differing_objects


    def initialize(self):
        raise NotImplementedError("Must be implemented by test subclass")


    def _normalize(self,*arrays):
        if any(a.size == 0 for a in arrays):
            return arrays
        norm = np.abs(arrays[0]).max()
        if norm == 0.:
            return arrays
        return (a / norm for a in arrays)


    def check_variable(self, var, atol=1e-8):
        v1, v2 = self.get_attribute(var)
        if v1 is None or v2 is None:
            print("Variable {} is None".format(var))
            return False
        if v1.ndim > 1:
            v1 = v1[2:-2, 2:-2, ...]
        if v2.ndim > 1:
            v2 = v2[2:-2, 2:-2, ...]
        passed = np.allclose(*self._normalize(v1,v2), atol=atol)
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
        self.initialize()
        flush()
        differing_scalars = self.check_scalar_objects()
        differing_arrays = self.check_array_objects()
        if differing_scalars or differing_arrays:
            print("Warning: the following attributes do not match between old and new pyom after initialization:")
            for s, (v1, v2) in differing_scalars.items():
                print("{}, {}, {}".format(s,v1,v2))
            for a, (v1, v2) in differing_arrays.items():
                print("{}, {}, {}".format(a,repr(np.asarray(v1).max()),repr(np.asarray(v2).max())))

        pyom_timers = {k: Timer("pyom " + k) for k in self.test_routines}
        pyom_legacy_timers = {k: Timer("pyom legacy " + k) for k in self.test_routines}
        all_passed = True
        for routine in self.test_routines.keys():
            pyom_routine, pyom_legacy_routine = self.get_routine(routine,self.test_module)
            pyom_args, pyom_legacy_args = self.test_routines[routine]
            with pyom_timers[routine]:
                pyom_routine(*pyom_args)
            pyom_timers[routine].printTime()
            with pyom_legacy_timers[routine]:
                pyom_legacy_routine(**pyom_legacy_args)
            pyom_legacy_timers[routine].printTime()
            passed = self.test_passed(routine)
            flush()
            if not passed:
                all_passed = False
                print("Test failed")
            self.initialize()
            flush()
        return all_passed
