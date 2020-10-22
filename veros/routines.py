import functools
import inspect
import threading
import contextlib

from loguru import logger

from veros.backend import flush


# stack helpers

class RoutineStack:
    def __init__(self):
        self.keep_full_stack = False
        self._stack = []
        self._current_idx = []

    @property
    def stack_level(self):
        return len(self._current_idx)

    def append(self, val):
        frame = self._stack
        for i in self._current_idx:
            frame = frame[i][1]

        self._current_idx.append(len(frame))
        frame.append([val, []])

    def pop(self):
        frame = self._stack
        for i in self._current_idx[:-1]:
            frame = frame[i][1]

        if self.keep_full_stack:
            last_val = frame[-1][0]
        else:
            last_val = frame.pop()[0]
        self._current_idx.pop()
        return last_val


@contextlib.contextmanager
def record_routine_stack():
    stack = CURRENT_CONTEXT.routine_stack
    if not stack.keep_full_stack:
        stack.keep_full_stack = True
        reset = True

    try:
        yield stack._stack
    finally:
        if reset:
            stack.keep_full_stack = False


@contextlib.contextmanager
def enter_routine(name, obj, timer, dist_safe=True):
    from veros import runtime_state as rst
    from veros.distributed import abort
    stack = CURRENT_CONTEXT.routine_stack

    logger.trace('{}> {}', '-' * stack.stack_level, name)
    stack.append(obj)

    reset_dist_safe = False
    if CURRENT_CONTEXT.is_dist_safe:
        if not dist_safe and rst.proc_num > 1:
            CURRENT_CONTEXT.is_dist_safe = False
            reset_dist_safe = True

    try:
        yield

    except:  # noqa: F722
        if reset_dist_safe:
            abort()
        raise

    finally:
        if reset_dist_safe:
            CURRENT_CONTEXT.is_dist_safe = True

        r = stack.pop()
        assert r is obj

        logger.trace('<{} {} ({:.3f}s)', '-' * stack.stack_level, name, timer.get_last_time())


# helper functions

def _get_func_name(function):
    return f'{inspect.getmodule(function).__name__}:{function.__qualname__}'


def _is_method(function):
    if inspect.ismethod(function):
        return True

    # hack for unbound methods: check if first argument is called "self"
    spec = inspect.getfullargspec(function)
    return spec.args and spec.args[0] == 'self'


def _deduplicate(somelist):
    return sorted(set(somelist))


# routine

def veros_routine(function=None, **kwargs):
    """
    .. note::

      This decorator should be applied to all functions that make use of the computational
      backend (even when subclassing :class:`veros.Veros`). The sole argument to the
      decorated function must be a VerosState instance.

    Example:
       >>> from veros import VerosSetup, veros_routine
       >>>
       >>> class MyModel(VerosSetup):
       >>>     @veros_routine
       >>>     def set_topography(self, vs):
       >>>         vs.kbot = np.random.randint(0, vs.nz, size=vs.kbot.shape)

    """
    # TODO: update docstring

    def inner_decorator(function):
        narg = 1 if _is_method(function) else 0
        num_params = len(inspect.signature(function).parameters)
        if narg >= num_params:
            raise TypeError('Veros routines must take at least one argument')

        routine_type_sig = inspect.signature(VerosRoutine)
        bound_args = routine_type_sig.bind(function, state_argnum=narg, **kwargs)
        bound_args.apply_defaults()

        routine = VerosRoutine(**bound_args.arguments)
        routine = functools.wraps(function)(routine)

        if routine.__doc__ is None:
            routine.__doc__ = ''

        routine.__doc__ += ''.join([
            f'\n\n    {key} = {val}' for key, val in bound_args.arguments.items()
        ])

        return routine

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class VerosRoutine:
    """Do not instantiate directly!"""

    def __init__(self, function, inputs=(), outputs=(), extra_outputs=(), settings=(), subroutines=(),
                 dist_safe=True, state_argnum=0):
        if isinstance(inputs, str):
            inputs = (inputs,)

        if isinstance(outputs, str):
            outputs = (outputs,)

        if isinstance(extra_outputs, str):
            extra_outputs = (extra_outputs,)

        if isinstance(settings, str):
            settings = (settings,)

        if isinstance(subroutines, VerosRoutine):
            subroutines = (subroutines,)

        self.function = function
        self.inputs = set(inputs)
        self.this_inputs = _deduplicate(inputs)
        self.outputs = set(outputs)
        self.this_outputs = _deduplicate(outputs)
        self.extra_outputs = _deduplicate(extra_outputs)
        self.settings = set(settings)
        self.subroutines = tuple(subroutines)

        for subroutine in subroutines:
            if not isinstance(subroutine, VerosRoutine):
                raise TypeError('Subroutines must themselves be veros routines')

            self.inputs |= set(subroutine.inputs)
            self.outputs |= set(subroutine.outputs)
            self.settings |= set(subroutine.settings)

        self.inputs = _deduplicate(self.inputs)
        self.outputs = _deduplicate(self.outputs)
        self.settings = _deduplicate(self.settings)

        self.dist_safe = dist_safe
        self.state_argnum = state_argnum

        self.name = _get_func_name(self.function)

    def __call__(self, *args, **kwargs):
        from veros import runtime_state as rst
        from veros.state import VerosStateBase, RestrictedVerosState

        orig_veros_state = args[self.state_argnum]

        if not isinstance(orig_veros_state, VerosStateBase):
            raise TypeError('first argument to a Veros routine must be a VerosState object')

        func_state = RestrictedVerosState(orig_veros_state)
        func_state._gather_arrays(self.inputs, flush=True)

        newargs = list(args)
        newargs[self.state_argnum] = func_state

        timer = orig_veros_state.profile_timers[self.name]

        with enter_routine(self.name, self, timer, dist_safe=self.dist_safe):
            execute = True
            if not CURRENT_CONTEXT.is_dist_safe:
                execute = rst.proc_rank == 0

            if execute:
                res = self.function(*newargs, **kwargs)

                if res is None:
                    res = {}

                if not isinstance(res, dict):
                    raise TypeError(f'Veros routines must return a single dict ({self.name})')

                if set(res.keys()) != set(self.this_outputs):
                    raise KeyError(
                        f'Veros routine {self.name} returned unexpected outputs '
                        f'(expected: {sorted(self.this_outputs)}, got: {sorted(res.keys())})'
                    )

            func_state._scatter_arrays(self.outputs, flush=True)

    def __get__(self, instance, instancetype):
        return functools.partial(self.__call__, instance)

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name} at {hex(id(self))}>'


# kernel

def veros_kernel(function=None, *, static_args=()):
    if isinstance(static_args, str):
        static_args = (static_args,)

    def inner_decorator(function):
        kernel = VerosKernel(function, static_args=static_args)
        kernel = functools.wraps(function)(kernel)
        return kernel

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class VerosKernel:
    """Do not instantiate directly!"""

    def __init__(self, function, static_args=()):
        """Do some parameter introspection and apply jax.jit"""
        from veros import runtime_settings

        # make sure function signature is in the form we need
        self.name = _get_func_name(function)
        self.func_sig = inspect.signature(function)
        func_params = self.func_sig.parameters

        allowed_param_types = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        if any(p.kind not in allowed_param_types for p in func_params.values()):
            raise ValueError(f'Veros kernels do not support *args, **kwargs, or keyword-only parameters ({self.name})')

        # parse static args
        func_argnames = list(func_params.keys())
        static_argnums = []
        for static_arg in static_args:
            try:
                arg_index = func_argnames.index(static_arg)
            except ValueError:
                raise ValueError(
                    f'Veros kernel {self.name} has no argument "{static_arg}", but it is given in static_args'
                ) from None

            static_argnums.append(arg_index)

        if runtime_settings.backend == 'jax':
            from jax import jit
            self.function = jit(function, static_argnums=static_argnums)
        else:
            self.function = function

    def run_with_state(self, veros_state, **kwargs):
        """Unpacks kernel arguments from given VerosState object and calls kernel."""
        from veros import runtime_settings

        # peel kernel parameters from veros state and given args
        func_params = self.func_sig.parameters

        func_kwargs = {
            arg: getattr(veros_state, arg)
            for arg in func_params.keys()
            if hasattr(veros_state, arg)
        }
        func_kwargs.update(kwargs)

        # when profiling, make sure all inputs are ready before starting the timer
        if runtime_settings.profile_mode:
            flush(list(func_kwargs.values()))

        timer = veros_state.profile_timers[self.name]

        with enter_routine(self.name, self.function, timer):
            out = self.__call__(**func_kwargs)

            if runtime_settings.profile_mode:
                out = flush(out)

        return out

    def __call__(self, *args, **kwargs):
        # JAX only accepts positional args when using static_argnums
        # so convert everything to positional for consistency
        bound_args = self.func_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self.function(*bound_args.arguments.values())

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name} at {hex(id(self))}>'


# TODO: remove
def run_kernel(func, vs, **kwargs):
    return func.run_with_state(vs, **kwargs)


# global context

CURRENT_CONTEXT = threading.local()
CURRENT_CONTEXT.is_dist_safe = True
CURRENT_CONTEXT.routine_stack = RoutineStack()
