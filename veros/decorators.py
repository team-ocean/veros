import functools
import signal
import inspect
import threading

from loguru import logger


CONTEXT = threading.local()
CONTEXT.is_dist_safe = True
CONTEXT.routine_stack = []
CONTEXT.stack_level = 0


def veros_routine(function=None, **kwargs):
    """
    .. note::

      This decorator should be applied to all functions that make use of the computational
      backend (even when subclassing :class:`veros.Veros`). The sole argument to the
      decorated function must be a Veros instance.

    Example:
       >>> from veros import VerosSetup, veros_routine
       >>>
       >>> class MyModel(VerosSetup):
       >>>     @veros_routine
       >>>     def set_topography(self):
       >>>         self.kbot[...] = np.random.randint(0, self.nz, size=self.kbot.shape)

    """
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


def _is_method(function):
    # spec = inspect.getfullargspec(function)
    # return spec.args and spec.args[0] == 'self'
    return inspect.ismethod(function)


class VerosRoutine:
    def __init__(self, function, inputs=(), outputs=(), extra_outputs=(), settings=(), subroutines=(), dist_safe=True, narg=0):
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
        self.this_inputs = set(inputs)
        self.outputs = set(outputs)
        self.this_outputs = set(outputs)
        self.extra_outputs = set(extra_outputs)
        self.settings = set(settings)
        self.subroutines = set(subroutines)

        for subroutine in subroutines:
            if not isinstance(subroutine, VerosRoutine):
                raise TypeError('Subroutines must themselves be veros routines')

            self.inputs |= subroutine.inputs
            self.outputs |= subroutine.outputs
            self.settings |= subroutine.settings

        self.dist_safe = dist_safe
        self.narg = narg

        self.name = f'{inspect.getmodule(self.function).__name__}:{self.function.__name__}'

        self.__call__ = functools.wraps(function)(
            functools.partial(VerosRoutine.__call__, self=self)
        )

    def __call__(self, *args, **kwargs):
        from veros import runtime_state as rst
        from veros.backend import flush
        from veros.state import VerosStateBase
        from veros.state_dist import DistributedVerosState

        logger.trace('{}> {}', '-' * CONTEXT.stack_level, self.name)
        CONTEXT.stack_level += 1
        CONTEXT.routine_stack.append(self)

        veros_state = args[self.narg]

        if not isinstance(veros_state, VerosStateBase):
            raise TypeError('first argument to a veros_routine must be a veros state object')

        reset_dist_safe = False
        if not CONTEXT.is_dist_safe:
            assert isinstance(veros_state, DistributedVerosState)
        elif not self.dist_safe and rst.proc_num > 1:
            reset_dist_safe = True

        if reset_dist_safe:
            dist_state = DistributedVerosState(veros_state)
            dist_state.gather_arrays(self.inputs)
            func_state = dist_state
            CONTEXT.is_dist_safe = False
        else:
            func_state = veros_state

        execute = True
        if not CONTEXT.is_dist_safe:
            execute = rst.proc_rank == 0

        newargs = list(args)
        newargs[self.narg] = func_state

        try:
            if execute:
                res = self.function(*newargs, **kwargs)
        except:  # noqa: F722
            if reset_dist_safe:
                CONTEXT.is_dist_safe = True
            raise
        else:
            if res is None:
                res = {}
            if not isinstance(res, dict):
                raise TypeError(f'Veros routines must return a single dict ({self.name})')
            if set(res.keys()) != self.this_outputs:
                raise KeyError(f'Veros routine {self.name} returned unexpected outputs (expected: {sorted(self.this_outputs)}, got: {sorted(res.keys())})')

            for key, val in res.items():
                if hasattr(func_state, key):
                    setattr(func_state, key, val)

            if reset_dist_safe:
                CONTEXT.is_dist_safe = True
                dist_state.scatter_arrays()
        finally:
            CONTEXT.stack_level -= 1
            logger.trace('<{} {}', '-' * CONTEXT.stack_level, self.name)
            r = CONTEXT.routine_stack.pop()
            assert r is self
            flush()

        return veros_state


def run_kernel(function, veros_state, **kwargs):
    """"""
    func_name = f'{inspect.getmodule(function).__name__}:{function.__qualname__}'

    # peel kernel parameters from veros state and given args
    func_params = inspect.signature(function).parameters

    func_kwargs = {
        arg: getattr(veros_state, arg)
        for arg in func_params.keys()
        if hasattr(veros_state, arg)
    }
    func_kwargs.update(kwargs)

    # check whether all required arguments are given
    for param in func_params.values():
        if param.name not in func_kwargs and param.default is inspect.Parameter.empty:
            raise TypeError(f'{func_name} missing required argument: {param.name}')

    func_args = (func_kwargs.get(param.name, param.default) for param in func_params.values())
    return function(*func_args)


def veros_kernel(function=None, static_args=()):
    if isinstance(static_args, str):
        static_args = (static_args,)

    def inner_decorator(function):
        """Do some parameter introspection and apply jax.jit"""
        from veros import runtime_settings

        func_name = f'{inspect.getmodule(function).__name__}:{function.__qualname__}'
        func_params = inspect.signature(function).parameters

        allowed_param_types = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        if any(p.kind not in allowed_param_types for p in func_params.values()):
            raise ValueError(f'Veros kernels do not support *args, **kwargs, or keyword-only parameters ({func_name})')

        func_argnames = list(func_params.keys())
        static_argnums = []
        for static_arg in static_args:
            try:
                arg_index = func_argnames.index(static_arg)
            except ValueError:
                raise ValueError(
                    f'Veros kernel {func_name} has no argument {static_arg}, but it is given in static_args'
                ) from None

            static_argnums.append(arg_index)

        if runtime_settings.backend == 'jax':
            from jax import jit
            return jit(function, static_argnums=static_argnums)

        return function

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


def dist_context_only(function):
    @functools.wraps(function)
    def dist_context_only_wrapper(*args, **kwargs):
        # args are assumed to be () or (arr, ...)
        from veros import runtime_state as rst

        if rst.proc_num == 1 or not CONTEXT.is_dist_safe:
            # no-op for sequential execution
            try:
                # return input array unchanged
                return args[0]
            except IndexError:
                return

        return function(*args, **kwargs)

    return dist_context_only_wrapper


def do_not_disturb(function):
    """Decorator that catches SIGINT and SIGTERM signals (e.g. after keyboard interrupt)
    and makes sure that the function body is executed before exiting.

    Useful e.g. for ensuring that output files are written properly.
    """
    signals = (signal.SIGINT, signal.SIGTERM)

    @functools.wraps(function)
    def dnd_wrapper(*args, **kwargs):
        old_handlers = {s: signal.getsignal(s) for s in signals}
        signal_received = {'sig': None, 'frame': None}

        def handler(sig, frame):
            if signal_received['sig'] is None:
                signal_received['sig'] = sig
                signal_received['frame'] = frame
                logger.error('Signal {} received - cleaning up before exit', sig)
            else:
                # force quit if more than one signal is received
                old_handlers[sig](sig, frame)

        for s in signals:
            signal.signal(s, handler)

        try:
            res = function(*args, **kwargs)

        finally:
            for s in signals:
                signal.signal(s, old_handlers[s])
            sig = signal_received['sig']
            if sig is not None:
                old_handlers[sig](signal_received['sig'], signal_received['frame'])

        return res

    return dnd_wrapper
