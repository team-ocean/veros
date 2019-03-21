from abc import abstractmethod, ABCMeta

from future.utils import with_metaclass


class LinearSolver(with_metaclass(ABCMeta)):
    @abstractmethod
    def __init__(self, vs):
        pass

    @abstractmethod
    def solve(self, vs, rhs, x0, boundary_val=None):
        pass
