from loguru import logger

BACKENDS = None


def init_environment():
    import os
    from . import runtime_state as rst

    if rst.proc_rank > 0:
        os.environ.update(
            BH_OPENMP_CACHE_READONLY='true',
            BH_UNSUP_WARN='false',
        )


def init_backends():
    init_environment()

    # populate available backend modules
    global BACKENDS
    BACKENDS = {}

    import numpy
    if numpy.__name__ == 'bohrium':
        logger.warning('Running veros with "python -m bohrium" is discouraged '
                       '(use "--backend bohrium" instead)')
        import numpy_force
        numpy = numpy_force

    BACKENDS['numpy'] = numpy

    try:
        import bohrium
    except ImportError:
        logger.warning('Could not import Bohrium (Bohrium backend will be unavailable)')
        BACKENDS['bohrium'] = None
    else:
        BACKENDS['bohrium'] = bohrium


def get_backend(backend_name):
    if BACKENDS is None:
        init_backends()

    if backend_name not in BACKENDS:
        raise ValueError('unrecognized backend {} (must be either of: {!r})'
                         .format(backend_name, list(BACKENDS.keys())))

    if BACKENDS[backend_name] is None:
        raise ValueError('backend "{}" failed to import'.format(backend_name))

    return BACKENDS[backend_name]


def get_vector_engine(np):
    from . import runtime_settings

    if runtime_settings.backend == 'bohrium':
        try:
            import bohrium_api
        except ImportError:
            return None

        if bohrium_api.stack_info.is_opencl_in_stack():
            return 'opencl'

        if bohrium_api.stack_info.is_cuda_in_stack():
            return 'cuda'

        return 'openmp'

    return None


def flush():
    from . import runtime_settings as rs

    if rs.backend == 'numpy':
        pass

    elif rs.backend == 'bohrium':
        get_backend(rs.backend).flush()

    else:
        raise RuntimeError('Unrecognized backend %s' % rs.backend)
