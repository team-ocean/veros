import threading

class wrapper:
    _safe_functions = ["__repr__", "__len__", "__str__"]

    def __init__(self, data):
        self._data = data
        self._writing = None

    def _wait_for_disk(self):
        if not self._writing == None:
            self._writing.join()
        self._writing = None

    def __getattr__(self, attr):
        origattr = self._data.__getattribute__(attr)
        if not callable(origattr) or attr in self._safe_functions:
            return origattr
        else:
            def funcwrap(*args, **kwargs):
                self._wait_for_disk()
                return wrapper(origattr(*args, **kwargs))
            return funcwrap

    def write(self):
        """ write shit to disk """
        print "I'm writing"

    def __getitem__(self, key):
        self._wait_for_disk()
        return self._data.__getitem__(key)

    def __coerce__(self, other):
        return None

    # Comparison functions
    def __cmp__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return self._data.__cmp__(other)

    def __eq__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return self._data == other

    def __ne__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return self._data != other

    def __le__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return self._data <= other

    def __ge__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return self._data >= other

    def __lt__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return self._data < other

    def __gt__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return self._data > other

    # Math functions
    def __add__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data + other)

    def __sub__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data - other)

    def __mul__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data * other)

    def __div__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data / other)

    def __floordiv__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data // other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __mod__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data % other)

    def __divmod__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(divmod(self._data, other))

    def __pow__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data ** other)

    def __lshift__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data << other)

    def __rshift__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data >> other)

    def __and__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data & other)

    def __or__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data | other)

    def __xor__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(self._data ^ other)

    def __radd__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other + self._data)

    def __rsub__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other - self._data)

    def __rmul__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other * self._data)

    def __rdiv__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other / self._data)

    def __rfloordiv__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other // self._data)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rmod__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other % self._data)

    def __rdivmod__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(divmod(other, self._data))

    def __rpow__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other ** self._data)

    def __rlshift__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other << self._data)

    def __rrshift__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other >> self._data)

    def __rand__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other & self._data)

    def __ror__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other | self._data)

    def __rxor__(self, other):
        if isinstance(other, wrapper):
            other = other._data
        return wrapper(other ^ self._data)

    # Inplace math operations
    def __iadd__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data += other
        return self

    def __isub__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data -= other
        return self

    def __imul__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data *= other
        return self

    def __idiv__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data /= other
        return self

    def __ifloordiv__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data //= other
        return self

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __imod__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data %= other
        return self

    def __ipow__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data **= other
        return self

    def __ilshift__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data <<= other
        return self

    def __irshift__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data >>= other
        return self

    def __iand__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data &= other
        return self

    def __ior__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data |= other
        return self

    def __ixor__(self, other):
        self._wait_for_disk()
        if isinstance(other, wrapper):
            other = other._data
        self._data ^= other
        return self
