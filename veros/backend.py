import numpy
import warnings

if numpy.__name__ == "bohrium":
    warnings.warn("Running veros with 'python -m bohrium' is discouraged "
                  "(use '--backend bohrium' instead)")
    import numpy_force
    numpy = numpy_force

try:
    import dask.array
except ImportError:
    warnings.warn("Could not import Dask")
    dask = None

try:
    import bohrium
    import bohrium.lapack
except ImportError:
    warnings.warn("Could not import Bohrium")
    bohrium = None

BACKENDS = {"numpy": numpy, "bohrium": bohrium, "dask": dask.array}


def get_backend(backend_name):
    if backend_name not in BACKENDS.keys():
        raise ValueError("unrecognized backend {} (must be either of: {!r})"
                         .format(backend_name, BACKENDS.keys()))
    if BACKENDS[backend_name] is None:
        raise ValueError("Backend '{}' failed to import".format(backend_name))
    return BACKENDS[backend_name], backend_name


def get_vector_engine(np):
    try:
        runtime_info = np.bh_info.runtime_info()
    except AttributeError:
        return None
    if "OpenCL" in runtime_info:
        return "opencl"
    if "CUDA" in runtime_info:
        return "cuda"
    return "openmp"


def flush(vs):
    if vs.backend_name == "numpy":
        pass

    elif vs.backend_name == "bohrium":
        vs.backend.flush()

    elif vs.backend_name == "dask":
        for variable in vs.variables:
            getattr(vs, variable).persist()

    else:
        raise RuntimeError("Unrecognized backend %r" % vs.backend_name)
