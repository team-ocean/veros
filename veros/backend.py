import warnings


# populate available backend modules
BACKENDS = {}

import numpy

if numpy.__name__ == "bohrium":
    warnings.warn("Running veros with 'python -m bohrium' is discouraged "
                    "(use '--backend bohrium' instead)")
    import numpy_force
    numpy = numpy_force

BACKENDS["numpy"] = numpy

try:
    import bohrium
except ImportError:
    warnings.warn("Could not import Bohrium (Bohrium backend will be unavailable)")
    BACKENDS["bohrium"] = None
else:
    BACKENDS["bohrium"] = bohrium


def get_backend(backend_name):
    if backend_name not in BACKENDS:
        raise ValueError("unrecognized backend {} (must be either of: {!r})"
                         .format(backend_name, list(BACKENDS.keys())))

    if BACKENDS[backend_name] is None:
        raise ValueError("backend '{}' failed to import".format(backend_name))

    return BACKENDS[backend_name]


def get_vector_engine(np):
    from veros import runtime_settings

    if runtime_settings.backend == "bohrium":
        try:
            import bohrium_api
        except ImportError:
            return None

        if bohrium_api.stack_info.is_opencl_in_stack():
            return "opencl"

        if bohrium_api.stack_info.is_cuda_in_stack():
            return "cuda"

        return "openmp"
    
    return None


def flush():
    from veros import runtime_settings as rs

    if rs.backend == "numpy":
        pass

    elif rs.backend == "bohrium":
        get_backend(rs.backend).flush()

    else:
        raise RuntimeError("Unrecognized backend %s" % rs.backend)
