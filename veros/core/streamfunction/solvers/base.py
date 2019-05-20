from abc import abstractmethod, ABCMeta


class LinearSolver(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, vs):
        pass

    @abstractmethod
    def solve(self, vs, rhs, x0, boundary_val=None):
        pass
