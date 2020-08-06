BACKENDS = None

BACKEND_MESSAGES = {
    'jax': 'Kernels are compiled during first iteration, be patient'
}


def init_environment():
    pass


def init_backends():
    init_environment()

    # populate available backend modules
    global BACKENDS
    BACKENDS = {}

    import numpy
    BACKENDS['numpy'] = numpy

    try:
        import jax
    except ImportError:
        jax = None

    BACKENDS['jax'] = jax


def get_backend(backend_name):
    if BACKENDS is None:
        init_backends()

    if backend_name not in BACKENDS:
        raise ValueError('unrecognized backend {} (must be either of: {!r})'
                         .format(backend_name, list(BACKENDS.keys())))

    if BACKENDS[backend_name] is None:
        raise ValueError('backend "{}" failed to import'.format(backend_name))

    return BACKENDS[backend_name]


# TODO: remove
def get_vector_engine(np):
    from veros import runtime_settings
    return None


# TODO: remove
def flush():
    pass
