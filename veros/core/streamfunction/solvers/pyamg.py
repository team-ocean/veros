import pyamg

from .scipy import SciPySolver
from .... import runtime_state as rst


class PyAMGSolver(SciPySolver):
    def __init__(self, vs):
        super(PyAMGSolver, self).__init__(vs)

        if rst.proc_rank == 0:
            ml = pyamg.smoothed_aggregation_solver(self._matrix)
            self._extra_args['M'] = ml.aspreconditioner()
