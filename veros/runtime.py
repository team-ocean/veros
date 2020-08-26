import os


def _default_mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    else:
        return MPI.COMM_WORLD


def twoints(v):
    return (int(v[0]), int(v[1]))


def parse_choice(choices, preserve_case=False):
    def validate(choice):
        if isinstance(choice, str) and not preserve_case:
            choice = choice.lower()

        if choice not in choices:
            raise ValueError('must be one of {}'.format(choices))

        return choice

    return validate


def parse_bool(obj):
    if not isinstance(obj, str):
        return bool(obj)

    return obj.lower() in {'1', 'true'}


LOGLEVELS = ('trace', 'debug', 'info', 'warning', 'error')
DEVICES = ('cpu', 'gpu', 'tpu')
FLOAT_TYPES = ('float64', 'float32')

AVAILABLE_SETTINGS = (
    # (name, type, default)
    ('backend', str, os.environ.get('VEROS_BACKEND', 'numpy')),
    ('device', parse_choice(DEVICES), os.environ.get('VEROS_DEVICE', 'cpu')),
    ('float_type', parse_choice(FLOAT_TYPES), os.environ.get('VEROS_FLOAT_TYPE', 'float64')),
    ('linear_solver', str, os.environ.get('VEROS_LINEAR_SOLVER', 'best')),
    ('num_proc', twoints, (1, 1)),
    ('profile_mode', parse_bool, os.environ.get('VEROS_PROFILE_MODE', '')),
    ('loglevel', parse_choice(LOGLEVELS), os.environ.get('VEROS_LOGLEVEL', 'info')),
    ('mpi_comm', None, _default_mpi_comm()),
    ('log_all_processes', parse_bool, os.environ.get('VEROS_LOG_ALL_PROCESSES', ''))
)


class RuntimeSettings:
    __slots__ = [
        '__locked__',
        '__setting_types__',
        '__settings__',
        *(setting for setting, _, _ in AVAILABLE_SETTINGS)
    ]

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
            try:
                val = stype(val)
            except (TypeError, ValueError) as e:
                raise ValueError('Got invalid value for runtime setting "{}": {}'.format(attr, str(e))) from None

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
        return backend.get_backend_module(runtime_settings.backend)

    def __setattr__(self, attr, val):
        raise TypeError('Cannot modify runtime state objects')
