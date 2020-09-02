import functools
import inspect
import threading
import contextlib

from loguru import logger


# stack helpers

class RoutineStack:
    def __init__(self):
        self.keep_full_stack = False
        self._stack = []
        self._current_idx = []

    @property
    def stack_level(self):
        return len(self._current_idx)

    def _get_current_frame(self):
        frame = self._stack
        for i in self._current_idx:
            frame = frame[i]
        return frame

    def append(self, val):
        self._get_current_frame().append(val)

    def pop(self):
        if self.keep_full_stack:
            last_val = self._get_current_frame()[-1]
        else:
            last_val = self._get_current_frame().pop()
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


def _flush(arrs):
    if isinstance(arrs, (list, tuple)):
        arr_iter = arrs
    else:
        arr_iter = [arrs]

    for arr in arr_iter:
        try:
            arr.block_until_ready()
        except AttributeError:
            pass

    return arrs


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
        return VerosRoutine(
            function, narg=narg, **kwargs
        )

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class VerosRoutine:
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
        newargs[self.narg] = func_state

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
        return functools.wraps(self.function)(
            functools.partial(self.__call__, instance)
        )


# kernel

def veros_kernel(function=None, *, static_args=()):
    if isinstance(static_args, str):
        static_args = (static_args,)

    def inner_decorator(function):
        """Do some parameter introspection and apply jax.jit"""
        from veros import runtime_settings

        # make sure function signature is in the form we need
        func_name = _get_func_name(function)
        func_sig = inspect.signature(function)
        func_params = func_sig.parameters

        allowed_param_types = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        if any(p.kind not in allowed_param_types for p in func_params.values()):
            raise ValueError(f'Veros kernels do not support *args, **kwargs, or keyword-only parameters ({func_name})')

        # parse static args
        func_argnames = list(func_params.keys())
        static_argnums = []
        for static_arg in static_args:
            try:
                arg_index = func_argnames.index(static_arg)
            except ValueError:
                raise ValueError(
                    f'Veros kernel {func_name} has no argument "{static_arg}", but it is given in static_args'
                ) from None

            static_argnums.append(arg_index)

        if runtime_settings.backend == 'jax':
            from jax import jit
            jitted_function = jit(function, static_argnums=static_argnums)

            @functools.wraps(function)
            def veros_kernel_wrapper(*args, **kwargs):
                # JAX only accepts positional args when using static_argnums
                bound_args = func_sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                return jitted_function(*bound_args.arguments.values())

            return veros_kernel_wrapper

        return function

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


def run_kernel(function, veros_state, **kwargs):
    """Unpacks kernel arguments from given VerosState object and calls kernel."""
    from veros import runtime_settings

    func_name = _get_func_name(function)

    # peel kernel parameters from veros state and given args
    func_params = inspect.signature(function).parameters

    func_kwargs = {
        arg: getattr(veros_state, arg)
        for arg in func_params.keys()
        if hasattr(veros_state, arg)
    }
    func_kwargs.update(kwargs)

    # when profiling, make sure all inputs are ready before starting the timer
    if runtime_settings.profile_mode:
        _flush(list(func_kwargs.values()))

    timer = veros_state.profile_timers[func_name]

    with enter_routine(func_name, function, timer):
        out = function(**func_kwargs)

        if runtime_settings.profile_mode:
            out = _flush(out)

    return out


# global context

CURRENT_CONTEXT = threading.local()
CURRENT_CONTEXT.is_dist_safe = True
CURRENT_CONTEXT.routine_stack = RoutineStack()
