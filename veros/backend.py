BACKENDS = None

BACKEND_MESSAGES = {
    'jax': 'Kernels are compiled during first iteration, be patient'
}


def init_environment():
    pass


def init_jax_config():
    import jax
    from veros import runtime_settings
    jax.config.enable_omnistaging()
    jax.config.update('jax_enable_x64', runtime_settings.float_type == 'float64')
    jax.config.update('jax_platform_name', runtime_settings.device)


def init_backends():
    init_environment()

    # populate available backend modules
    global BACKENDS
    BACKENDS = {}

    import numpy
    BACKENDS['numpy'] = numpy

    try:
        import jax  # noqa: F401
    except ImportError:
        jnp = None
    else:
        init_jax_config()
        import jax.numpy as jnp

    BACKENDS['jax'] = jnp


def get_backend_module(backend_name):
    if BACKENDS is None:
        init_backends()

    if backend_name not in BACKENDS:
        raise ValueError('unrecognized backend {} (must be either of: {!r})'
                         .format(backend_name, list(BACKENDS.keys())))

    if BACKENDS[backend_name] is None:
        raise ValueError('backend "{}" failed to import'.format(backend_name))

    return BACKENDS[backend_name]


def get_curent_device_name():
    from veros import runtime_settings
    if runtime_settings.backend != 'jax':
        return 'cpu'

    return runtime_settings.device
