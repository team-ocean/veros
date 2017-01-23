class ShiftableSlice(object):
    """
    A class that returns a shifted slice when it is added to an integer.
    """
    def __init__(self,start=None,stop=None,step=None):
        self.start = start
        self.stop = stop
        self.step = step

    def _shift_start(self,other):
        start = self.start
        if start is None:
            if other >= 0:
                return other or None
            else:
                raise ValueError("Index underflow")
        result = start + other
        if start > 0 and result < 0:
            raise ValueError("Index underflow")
        return result or None

    def _shift_stop(self,other):
        stop = self.stop
        if stop is None:
            if other <= 0:
                return other or None
            else:
                raise ValueError("Index overflow")
        result = stop + other
        if stop < 0 and result > 0:
            raise ValueError("Index overflow")
        return result or None

    def __add__(self,other):
        other = int(other)
        return slice(self._shift_start(other), self._shift_stop(other), self.step)

    def __sub__(self,other):
        return self.__add__(-other)

def make_slice(start=None, stop=None, step=None):
    """
    Returns a matching pair of slice and ShiftableSlice for convenience.
    """
    return slice(start,stop,step), ShiftableSlice(start, stop, step)
