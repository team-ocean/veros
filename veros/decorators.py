import functools
import signal
import threading
import logging
import inspect

CONTEXT = threading.local()
CONTEXT.is_dist_safe = True
CONTEXT.wrapped_methods = []


def veros_method(function=None, **kwargs):
    """Decorator that injects the current backend as variable ``np`` into the wrapped function.

    .. note::

      This decorator should be applied to all functions that make use of the computational
      backend (even when subclassing :class:`veros.Veros`). The first argument to the
      decorated function must be a Veros instance.

    Example:
       >>> from veros import Veros, veros_method
       >>> 
       >>> class MyModel(Veros):
       >>>     @veros_method
       >>>     def set_topography(self):
       >>>         self.kbot[...] = np.random.randint(0, self.nz, size=self.kbot.shape)

    """
    if function is not None:
        narg = 1 if _is_method(function) else 0
        return _veros_method(function, narg=narg)

    flush_on_exit = not kwargs.pop("inline", False)
    dist_safe = kwargs.pop("dist_safe", True)

    if not dist_safe and "local_variables" not in kwargs:
        raise ValueError("local_variables argument must be given if dist_safe=False")

    local_vars = kwargs.pop("local_variables", [])
    dist_only = kwargs.pop("dist_only", False)

    def inner_decorator(function):
        narg = 1 if _is_method(function) else 0
        return _veros_method(
            function, flush_on_exit=flush_on_exit, narg=narg,
            dist_safe=dist_safe, local_vars=local_vars, dist_only=dist_only
        )

    return inner_decorator


def _is_method(function):
    spec = inspect.getargspec(function)
    return spec.args and spec.args[0] == 'self'


def _veros_method(function, flush_on_exit=True, dist_safe=True, local_vars=None,
                  dist_only=False, narg=0):
    from . import runtime_settings as rs, runtime_state as rst
    from .backend import flush, get_backend
    from .state import VerosState
    from .state_dist import DistributedVerosState
    CONTEXT.wrapped_methods.append(function)

    @functools.wraps(function)
    def veros_method_wrapper(*args, **kwargs):
        from .distributed import broadcast

        veros_state = args[narg]

        if not isinstance(veros_state, VerosState):
            raise TypeError("first argument to a veros_method must be subclass of VerosState")

        reset_dist_safe = False
        if not CONTEXT.is_dist_safe:
            assert isinstance(veros_state, DistributedVerosState)
        elif not dist_safe and rst.proc_num > 1:
            reset_dist_safe = True

        if reset_dist_safe:
            dist_state = DistributedVerosState(veros_state)
            dist_state.gather_arrays(local_vars)
            func_state = dist_state
            CONTEXT.is_dist_safe = False
        else:
            func_state = veros_state

        execute = True
        if not CONTEXT.is_dist_safe:
            execute = rst.proc_rank == 0

        g = function.__globals__
        sentinel = object()

        oldvalue = g.get('np', sentinel)
        g['np'] = get_backend(rs.backend)

        newargs = list(args)
        newargs[narg] = func_state

        res = None
        try:
            if execute:
                res = function(*newargs, **kwargs)
        except:
            if reset_dist_safe:
                CONTEXT.is_dist_safe = True
            raise
        else:
            if reset_dist_safe:
                CONTEXT.is_dist_safe = True
                res = broadcast(veros_state, res)
                dist_state.scatter_arrays()
        finally:
            if oldvalue is sentinel:
                del g['np']
            else:
                g['np'] = oldvalue

            if flush_on_exit:
                flush()

        return res

    return veros_method_wrapper


_veros_method.methods = []


def dist_context_only(function):
    @functools.wraps(function)
    def dist_context_only_wrapper(vs, arr, *args, **kwargs):
        from . import runtime_state as rst

        if rst.proc_num == 1 or not CONTEXT.is_dist_safe:
            # no-op for sequential execution
            return arr

        return function(vs, arr, *args, **kwargs)

    return dist_context_only_wrapper


def do_not_disturb(function):
    """Decorator that catches SIGINT and SIGTERM signals (e.g. after keyboard interrupt)
    and makes sure that the function body is executed before exiting.

    Useful e.g. for ensuring that output files are written properly.
    """
    signals = (signal.SIGINT, signal.SIGTERM)

    @functools.wraps(function)
    def dnd_wrapper(*args, **kwargs):
        signal_received = {"sig": None, "frame": None}

        def handler(sig, frame):
            signal_received["sig"] = sig
            signal_received["frame"] = frame
            logging.error("{} received - cleaning up before exit".format(sig))

        old_handlers = {s: signal.getsignal(s) for s in signals}
        for s in signals:
            signal.signal(s, handler)

        try:
            res = function(*args, **kwargs)

        finally:
            for s in signals:
                signal.signal(s, old_handlers[s])
            sig = signal_received["sig"]
            if sig is not None:
                old_handlers[sig](signal_received["sig"], signal_received["frame"])

        return res

    return dnd_wrapper
