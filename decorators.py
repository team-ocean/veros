import functools
import signal
import logging


def veros_class_method(function):
    return _veros_method(function, True, 1)


def veros_method(function):
    """Decorator that injects the current backend as variable ``np`` into the wrapped function.

    .. note::

      This decorator should be applied to all functions that make use of the computational
      backend (even when subclassing :class:`climate.veros.Veros`). The first argument to the
      decorated function must be a Veros instance.

    Example:
       >>> from climate.veros import Veros, veros_method
       >>>
       >>> class MyModel(Veros):
       >>>     @veros_method
       >>>     def set_topography(self):
       >>>         self.kbot[...] = np.random.randint(0, self.nz, size=self.kbot.shape)
    """
    return _veros_method(function, True)


def veros_inline_method(function):
    return _veros_method(function, False)


def _veros_method(function, flush_on_exit, narg=0):
    import veros
    _veros_method.methods.append(function)

    @functools.wraps(function)
    def veros_method_wrapper(*args, **kwargs):
        veros_instance = args[narg]
        if not isinstance(veros_instance, veros.Veros):
            raise TypeError("first argument to a veros_method must be subclass of Veros")
        g = function.__globals__
        sentinel = object()

        oldvalue = g.get('np', sentinel)
        g['np'] = veros_instance.backend

        try:
            res = function(*args, **kwargs)
        finally:
            if oldvalue is sentinel:
                del g['np']
            else:
                g['np'] = oldvalue
        if flush_on_exit:
            veros_instance.flush()
        return res
    return veros_method_wrapper


_veros_method.methods = []


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
            if not sig is None:
                old_handlers[sig](signal_received["sig"], signal_received["frame"])

        return res

    return dnd_wrapper
