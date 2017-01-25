import numpy as np

def _get_fortran_attribute(pyom,attribute):
    val = None
    for module in ("main","isoneutral","idemix","tke","eke"):
        module_path = module + "_module"
        module_handle = getattr(pyom, module_path)
        try:
            val2 = getattr(module_handle, v)
        except AttributeError:
            pass
    return val

def _get_python_attribute(pyom,attribute):
    try:
        return getattr(pyom,attribute)
    except AttributeError:
        return None

def check_scalar_objects(pyom1,pyom2):
    differing_objects = {}
    pyom1_getattr = _get_fortran_attribute if pyom1.legacy_mode or False else _get_python_attribute
    pyom2_getattr = _get_fortran_attribute if pyom2.legacy_mode or False else _get_python_attribute
    with open("scalar_attributes","r") as f:
        for v in f:
            v = v.strip()
            val1 = pyom1_getattr(pyom1,v)
            val2 = pyom2_getattr(pyom2,v)
            if ((val1 is None) != (val2 is None)) or val1 != val2:
                differing_objects[v] = (val1, val2)
    return differing_objects

def check_array_objects(pyom1,pyom2):
    differing_objects = {}
    pyom1_getattr = _get_fortran_attribute if pyom1.legacy_mode or False else _get_python_attribute
    pyom2_getattr = _get_fortran_attribute if pyom2.legacy_mode or False else _get_python_attribute
    with open("array_attributes","r") as f:
        for v in f:
            v = v.strip()
            try:
                val1 = getattr(sim_new,v)
                if val1.ndim >= 2:
                    val1 = val1[2:sim_new.nx+2,2:sim_new.ny+2,...]
            except AttributeError:
                val1 = None
            val2 = None
            for module in ("main","isoneutral","idemix","tke","eke"):
                module_path = module + "_module"
                    val2 = getattr(module_handle, v.lower())
                    if val2.ndim >= 2:
                        val2 = val2[2:-2,2:-2,...]
                except AttributeError:
                    pass
            if not (val1 is None and val2 is None) and not np.array_equal(val1,val2):
                print(v)
                try:
                    diff = np.abs(val1-val2)
                except TypeError:
                    print(type(val1),type(val2))
                    continue
                close = np.allclose(val1,val2)
                print(diff.max(), np.abs(val1).max(), np.abs(val2).max(), close)
                if not close:
                    print(val1.shape, np.count_nonzero(val1 != val2), np.nonzero(val1 != val2))
