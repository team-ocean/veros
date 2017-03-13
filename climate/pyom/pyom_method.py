from functools import wraps

def pyom_method(function):
    """Decorator that injects the current backend as variable ``np`` into the wrapped function.

    .. note::

      This decorator should be applied to all functions that make use of the computational
      backend (even when subclassing :class:`climate.pyom.PyOM`).

    Example:
       >>> from climate.pyom import PyOM, pyom_method
       >>>
       >>> class MyModel(PyOM):
       >>>     @pyom_method
       >>>     def set_topography(self):
       >>>         self.kbot[...] = np.random.randint(0, self.nz, size=self.kbot.shape)
    """
    pyom_method.methods.append(function)
    @wraps(function)
    def wrapper(pyom, *args, **kwargs):
        if not isinstance(pyom, PyOM):
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
        return res
    return wrapper
pyom_method.methods = []
