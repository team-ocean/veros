from functools import wraps

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

def _veros_method(function, flush_on_exit):
    import veros
    _veros_method.methods.append(function)
    @wraps(function)
    def veros_method_wrapper(veros_instance, *args, **kwargs):
        if not isinstance(veros_instance, veros.Veros):
            raise TypeError("first argument to a veros_method must be subclass of Veros")
        g = function.__globals__
        sentinel = object()

        oldvalue = g.get('np', sentinel)
        g['np'] = veros_instance.backend

        try:
            res = function(veros_instance, *args, **kwargs)
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
