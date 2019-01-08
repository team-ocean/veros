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
        backend_modules["bohrium"] = bohrium
    except ImportError:
        warnings.warn("Could not import Bohrium")

    if backend_modules[backend_name] is None:
        raise ValueError("Backend '{}' failed to import".format(backend_name))
    return backend_modules[backend_name], backend_name


def get_vector_engine(backend, backend_name):
    if backend_name == "bohrium":
        import bohrium_api
        if bohrium_api.stack_info.is_opencl_in_stack():
            return "opencl"
        if bohrium_api.stack_info.is_cuda_in_stack():
            return "cuda"
        return "openmp"
    
    return None


def flush(vs):
    if vs.backend_name == "numpy":
        pass

    elif vs.backend_name == "bohrium":
        vs.backend.flush()

    else:
        raise RuntimeError("Unrecognized backend %r" % vs.backend_name)
