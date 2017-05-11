import numpy
import warnings

if numpy.__name__ == "bohrium":
    warnings.warn("Running veros with 'python -m bohrium' is discouraged (use '--backend bohrium' instead)")
    import numpy_force
    numpy = numpy_force

try:
    import bohrium
    import bohrium.lapack
except ImportError:
    warnings.warn("Could not import Bohrium")
    bohrium = None

BACKENDS = {"numpy": numpy, "bohrium": bohrium}


def get_backend(backend):
    if backend not in BACKENDS.keys():
        raise ValueError("unrecognized backend {} (must be either of: {!r})"
                         .format(backend, BACKENDS.keys()))
    if BACKENDS[backend] is None:
        raise ValueError("{} backend failed to import".format(backend))
    return BACKENDS[backend], backend
