import sys
import os
#import matplotlib.pyplot as plt
import numpy as np
import bohrium as bh
from collections import OrderedDict

from veros import VerosLegacy, variables
from veros.timer import Timer

flush = bh.flush

class VerosTest(object):
    legacy_modules = ("main_module", "isoneutral_module", "tke_module",
                      "eke_module", "idemix_module")
    array_attribute_file = os.path.join(os.path.dirname(__file__), "array_attributes")
    scalar_attribute_file = os.path.join(os.path.dirname(__file__), "scalar_attributes")
    extra_settings = None
    test_module = None
    test_routines = None

    def __init__(self, dims=None, fortran=None):
        self.veros_new = VerosLegacy()
        self.veros_new.pyom_compatibility_mode = True
        if not fortran:
            try:
                fortran = sys.argv[1]
            except IndexError:
                raise RuntimeError("Path to fortran library must be given via keyword argument or command line")
        self.veros_legacy = VerosLegacy(fortran=fortran)

        if dims:
            self.nx, self.ny, self.nz = dims
        self.set_attribute("nx", self.nx)
        self.set_attribute("ny", self.ny)
        self.set_attribute("nz", self.nz)
        if self.extra_settings:
            for attribute, value in self.extra_settings.items():
                self.set_attribute(attribute, value)
        self.veros_new.set_legacy_parameter()
        variables.allocate_variables(self.veros_new)
        self.veros_legacy.fortran.my_mpi_init(0)
        self.veros_legacy.fortran.pe_decomposition()
        self.veros_legacy.set_legacy_parameter()
        self.veros_legacy.fortran.allocate_main_module()
        self.veros_legacy.fortran.allocate_isoneutral_module()
        self.veros_legacy.fortran.allocate_tke_module()
        self.veros_legacy.fortran.allocate_eke_module()
        self.veros_legacy.fortran.allocate_idemix_module()


    def set_attribute(self, attribute, value):
        if isinstance(value, np.ndarray):
            getattr(self.veros_new, attribute)[...] = value
        else:
            setattr(self.veros_new, attribute, value)
        for module in self.legacy_modules:
            module_handle = getattr(self.veros_legacy,module)
            if hasattr(module_handle, attribute):
                try:
                    v = np.asfortranarray(value).copy2numpy()
                except AttributeError:
                    v = np.asfortranarray(value)
                setattr(module_handle, attribute, v)
                assert np.all(value == getattr(module_handle, attribute)), attribute
                return
        raise AttributeError("Legacy pyOM has no attribute {}".format(attribute))


    def get_attribute(self, attribute):
        try:
            veros_attr = getattr(self.veros_new, attribute)
        except AttributeError:
            veros_attr = None
        try:
            veros_attr = veros_attr.copy2numpy()
        except AttributeError:
            pass
        veros_legacy_attr = None
        for module in self.legacy_modules:
            module_handle = getattr(self.veros_legacy,module)
            if hasattr(module_handle, attribute):
                veros_legacy_attr = getattr(module_handle, attribute)
        return veros_attr, veros_legacy_attr


    def get_routine(self, routine, submodule=None):
        if submodule:
            veros_module_handle = submodule
        else:
            veros_module_handle = self.veros_new
        veros_routine = getattr(veros_module_handle, routine)
        veros_legacy_routine = getattr(self.veros_legacy.fortran, routine)
        return veros_routine, veros_legacy_routine


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


    def check_variable(self, var, atol=1e-8, data=None):
        if data is None:
            v1, v2 = self.get_attribute(var)
        else:
            v1, v2 = data
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
            #if v1.ndim == 2:
                #fig, axes = plt.subplots(1,3)
                #axes[0].imshow(v1)
                #axes[0].set_title("New")
                #axes[1].imshow(v2)
                #axes[1].set_title("Legacy")
                #axes[2].imshow(v1 - v2)
                #axes[2].set_title("diff")
                #fig.suptitle(var)
        return passed

    def run(self):
        self.initialize()
        flush()
        differing_scalars = self.check_scalar_objects()
        differing_arrays = self.check_array_objects()
        if differing_scalars or differing_arrays:
            print("Warning: the following attributes do not match between old and new veros after initialization:")
            for s, (v1, v2) in differing_scalars.items():
                print("{}, {}, {}".format(s,v1,v2))
            for a, (v1, v2) in differing_arrays.items():
                if np.asarray(v1).size == 0:
                    print("{}, {!r}, {!r}".format(a, None, np.max(v2)))
                elif np.asarray(v2).size == 0:
                    print("{}, {!r}, {!r}".format(a, np.max(v1), None))
                else:
                    print("{}, {!r}, {!r}".format(a,np.max(v1),np.max(v2)))

        veros_timers = {k: Timer("veros " + k) for k in self.test_routines}
        veros_legacy_timers = {k: Timer("veros legacy " + k) for k in self.test_routines}
        all_passed = True
        for routine in self.test_routines.keys():
            veros_routine, veros_legacy_routine = self.get_routine(routine,self.test_module)
            veros_args, veros_legacy_args = self.test_routines[routine]
            with veros_timers[routine]:
                veros_routine(*veros_args)
            veros_timers[routine].printTime()
            with veros_legacy_timers[routine]:
                veros_legacy_routine(**veros_legacy_args)
            veros_legacy_timers[routine].printTime()
            passed = self.test_passed(routine)
            flush()
            if not passed:
                all_passed = False
                print("Test failed")
            self.initialize()
            flush()
        return all_passed
