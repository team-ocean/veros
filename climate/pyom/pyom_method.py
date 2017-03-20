from functools import wraps

import climate.pyom

def pyom_method(function):
    """Decorator that injects the current backend as variable ``np`` into the wrapped function.

    .. note::

      This decorator should be applied to all functions that make use of the computational
      backend (even when subclassing :class:`climate.pyom.PyOM`). The first argument to the
      decorated function must be a PyOM instance.

    Example:
       >>> from climate.pyom import PyOM, pyom_method
       >>>
       >>> class MyModel(PyOM):
       >>>     @pyom_method
       >>>     def set_topography(self):
       >>>         self.kbot[...] = np.random.randint(0, self.nz, size=self.kbot.shape)
    """
    return _pyom_method(function, True)

def pyom_inline_method(function):
    return _pyom_method(function, False)

def _pyom_method(function, flush_on_exit):
    _pyom_method.methods.append(function)
    @wraps(function)
    def pyom_method_wrapper(pyom, *args, **kwargs):
        if not isinstance(pyom, climate.pyom.PyOM):
            raise TypeError("first argument to a pyom_method must be subclass of PyOM")
        g = function.__globals__
        sentinel = object()

        oldvalue = g.get('np', sentinel)
        g['np'] = pyom.backend

        try:
            res = function(pyom, *args, **kwargs)
        finally:
            if oldvalue is sentinel:
                del g['np']
            else:
                g['np'] = oldvalue
        if flush_on_exit:
            pyom.flush()
        return res
    return pyom_method_wrapper
_pyom_method.methods = []
