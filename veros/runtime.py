def _default_mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    else:
        return MPI.COMM_WORLD


def twoints(v):
    return (int(v[0]), int(v[1]))


def loglevel(v):
    loglevels = ('trace', 'debug', 'info', 'warning', 'error')
    if v not in loglevels:
        raise ValueError('loglevel must be one of %r' % loglevels)
    return v


AVAILABLE_SETTINGS = (
    # (name, type, default)
    ('backend', str, 'numpy'),
    ('linear_solver', str, 'best'),
    ('num_proc', twoints, (1, 1)),
    ('profile_mode', bool, False),
    ('loglevel', loglevel, 'info'),
    ('mpi_comm', None, _default_mpi_comm())
)


class RuntimeSettings:
    def __init__(self):
        self.__locked__ = False
        self.__setting_types__ = {}

        for setting, typ, default in AVAILABLE_SETTINGS:
            setattr(self, setting, default)
            self.__setting_types__[setting] = typ

        self.__settings__ = set(self.__setting_types__.keys())
        self.__locked__ = True

    def __setattr__(self, attr, val):
        if attr == '__locked__' or not self.__locked__:
            return super(RuntimeSettings, self).__setattr__(attr, val)

        # prevent adding new settings
        if attr not in self.__settings__:
            raise AttributeError('Unknown runtime setting %s' % attr)

        # coerce type
        stype = self.__setting_types__.get(attr)
        if stype is not None:
            val = stype(val)

        return super(RuntimeSettings, self).__setattr__(attr, val)

    def __repr__(self):
        setval = ', '.join(
            '%s=%s' % (key, repr(getattr(self, key))) for key in self.__settings__
        )
        return '{clsname}({setval})'.format(
            clsname=self.__class__.__name__,
            setval=setval
        )


class RuntimeState:
    """Unifies attributes from various modules in a simple read-only object"""

    __slots__ = []

    @property
    def proc_rank(self):
        from . import runtime_settings
        comm = runtime_settings.mpi_comm

        if comm is None:
            return 0

        return comm.Get_rank()

    @property
    def proc_num(self):
        from . import runtime_settings
        comm = runtime_settings.mpi_comm

        if comm is None:
            return 1

        return comm.Get_size()

    @property
    def proc_idx(self):
        from . import distributed
        return distributed.proc_rank_to_index(self.proc_rank)

    @property
    def backend_module(self):
        from . import backend, runtime_settings
        return backend.get_backend(runtime_settings.backend)

    @property
    def vector_engine(self):
        from . import backend
        return backend.get_vector_engine(self.backend_module)

    def __setattr__(self, attr, val):
        raise TypeError('Cannot modify runtime state objects')
