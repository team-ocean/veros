BACKENDS = ["numpy", "bohrium"]


def get_backend(backend_name):
    if backend_name not in BACKENDS:
        raise ValueError("unrecognized backend {} (must be either of: {!r})"
                         .format(backend_name, BACKENDS))

    backend_modules = {backend: None for backend in BACKENDS}

    import numpy
    import warnings

    if numpy.__name__ == "bohrium":
        warnings.warn("Running veros with 'python -m bohrium' is discouraged "
                      "(use '--backend bohrium' instead)")
        import numpy_force
        numpy = numpy_force

    backend_modules["numpy"] = numpy

    try:
        import bohrium
        import bohrium.lapack
        backend_modules["bohrium"] = bohrium
    except ImportError:
        warnings.warn("Could not import Bohrium")

    if backend_modules[backend_name] is None:
        raise ValueError("Backend '{}' failed to import".format(backend_name))
    return backend_modules[backend_name], backend_name


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

    else:
        raise RuntimeError("Unrecognized backend %r" % vs.backend_name)
